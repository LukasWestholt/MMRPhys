'''
Remote Vital Signs Detection from Video
This script demonstrates how to estimate vital signs (Heart Rate and Respiration Rate) from a video file.
The script uses a pre-trained model to estimate vital signs from facial video data.

Usage:
    python infer_from_video.py --config config.yaml
'''

import torch
import onnxruntime as ort
from pathlib import Path
import numpy as np
from scipy import signal
from collections import OrderedDict
import collections
from scipy.signal import periodogram, welch
import threading
from queue import Queue


class SignalProcessor:
    def __init__(self, fs):
        self.fs = fs
        self.init_filters()

    def init_filters(self):
        lowcut_bvp = 0.6
        highcut_bvp = 3.3
        lowcut_rsp = 0.1
        highcut_rsp = 0.5
        order = 2
        nyquist = 0.5 * self.fs
        low_bvp = lowcut_bvp / nyquist
        high_bvp = highcut_bvp / nyquist
        low_rsp = lowcut_rsp / nyquist
        high_rsp = highcut_rsp / nyquist
        self.b_bvp, self.a_bvp = signal.butter(
            order, [low_bvp, high_bvp], btype='band')
        self.b_rsp, self.a_rsp = signal.butter(
            order, [low_rsp, high_rsp], btype='band')

    def next_power_of_2(self, x):
        """Calculate the nearest power of 2."""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def calculate_fft_rate(self, phys_signal, low_pass=0.6, high_pass=3.3):
        """Calculate heart rate/ resp rate using Fast Fourier transform (FFT)."""
        phys_signal = np.expand_dims(phys_signal, 0)
        N = phys_signal.shape[1]
        if N <= 30*self.fs:
            nfft = self.next_power_of_2(N)
            f_phys, pxx_phys = periodogram(
                phys_signal, fs=self.fs, nfft=nfft, detrend=False)
        else:
            f_phys, pxx_phys = welch(phys_signal, fs=self.fs, nperseg=N//2, detrend=False)
        fmask_phys = np.argwhere((f_phys >= low_pass) & (f_phys <= high_pass))
        mask_phys = np.take(f_phys, fmask_phys)
        mask_pxx = np.take(pxx_phys, fmask_phys)
        fft_hr = np.take(mask_phys, np.argmax(mask_pxx, 0))[0] * 60
        return fft_hr

    def post_process(self, bvp, rsp):
        # apply normalization and bandpass filter
        bvp = (bvp - np.mean(bvp))/np.std(bvp)
        rsp = (rsp - np.mean(rsp))/np.std(rsp)
        bvp = signal.filtfilt(self.b_bvp, self.a_bvp, bvp)
        rsp = signal.filtfilt(self.b_rsp, self.a_rsp, rsp)
        return bvp, rsp

    def compute_metrics(self, bvp, rsp):
        # compute heart rate
        hr = self.calculate_fft_rate(bvp, low_pass=0.6, high_pass=3.3)
        # validate heart rate
        if hr < 40 or hr > 200:
            hr = np.nan
        else:
            hr = np.round(hr, 0)
        print(f'Heart rate: {hr}')

        # Compute respiration rate
        rr = self.calculate_fft_rate(rsp, low_pass=0.1, high_pass=0.5)
        # validate respiration rate
        if rr < 5 or rr > 40:
            rr = np.nan
        else:
            rr = np.round(rr, 0)
        print(f'Resp rate: {rr}')
        return hr, rr


class InferenceWorker:
    def __init__(self, config):
        self.model_path = Path(config['model']['path'])
        self.model_type = config['model']['type']
        self.num_frames = config['model']['input_shape']['num_frames']
        self.in_channels = config['model']['input_shape']['channels']
        self.height = config['model']['input_shape']['height']
        self.width = config['model']['input_shape']['width']
        self.fs = config['video']['sampling_rate']

        assert self.height == self.width, "Height and width must be equal"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.init_model()


    def init_model(self):
        self.md_config = {}
        self.md_config["FRAME_NUM"] = self.num_frames
        self.md_config["TASKS"] = ["BVP", "RSP"]

        if self.model_type == 'torch':
            self.init_torch_model()
        elif self.model_type == 'onnx':
            self.init_onnx_model()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def init_onnx_model(self):
        from mmrphys.tools.torch2onnx.MMRPhysSEF import MMRPhysSEF as rPhysModel

        self.model = rPhysModel(self.num_frames, self.md_config, self.in_channels)
        self.model = ort.InferenceSession(self.model_path)


    def init_torch_model(self):
        if self.height == 9:
            from mmrphys.neural_methods.model.MMRPhys.MMRPhysSEF import MMRPhysSEF as rPhysModel
        elif self.height == 36:
            from mmrphys.neural_methods.model.MMRPhys.MMRPhysMEF import MMRPhysMEF as rPhysModel
        elif self.height == 72:
            from mmrphys.neural_methods.model.MMRPhys.MMRPhysLEF import MMRPhysLEF as rPhysModel
        else:
            raise ValueError(f"Invalid height and width: {self.height, self.width}")

        self.model = rPhysModel(self.num_frames, self.md_config, self.in_channels)
        model_weights = torch.load(self.model_path, map_location=self.device)
        model_weights = self.map_weights(model_weights)

        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(model_weights)
        self.model.to(self.device)
        self.model.eval()


    def map_weights(self, old_state_dict):
        """Maps weights from old Sequential ConvBlock3D to new implementation"""
        new_state_dict = OrderedDict()

        for key, value in old_state_dict.items():
            # Skip unnecessary weights
            if any(skip in key for skip in ['fsam', 'bias1']):
                continue
            else:
                new_state_dict[key] = value

        return new_state_dict


    def infer_rphys(self, frame_buffer):
        if self.model_type == 'onnx':
            if self.model.get_inputs()[0].type == "tensor(float16)":
                return self.model.run(None, {'input': frame_buffer.astype(np.float16)})
            else:
                print(f"Assume {self.model_type} is float32.")
                return self.model.run(None, {'input': frame_buffer.astype(np.float32)})
        else:
            with torch.no_grad():
                torch_frames = torch.from_numpy(frame_buffer).float().to(self.device)
                output = self.model(torch_frames)
                bvp = output[0].cpu().numpy()
                rsp = output[1].cpu().numpy()
                return bvp, rsp


class RemoteVitalSigns:
    def __init__(self, config):
        self.model_path = Path(config['model']['path'])
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Initialize all parameters from config
        self.model_type = config['model']['type']
        self.num_frames = config['model']['input_shape']['num_frames']
        self.in_channels = config['model']['input_shape']['channels']
        self.height = config['model']['input_shape']['height']
        self.width = config['model']['input_shape']['width']
        self.fs = config['video']['sampling_rate']
        self.inference_interval = int(config['processing']['inference_interval'])
        self.plot_duration = config['processing']['plot_duration']

        # Initialize components
        self.signal_processor = SignalProcessor(self.fs)
        self.model_instance = InferenceWorker(config)

        # Initialize buffers
        self.init_buffers()

        # Initialize threading components
        self.frame_queue = Queue(maxsize=100)
        self.result_queue = Queue()
        self.stop_event = threading.Event()

        # Additional buffer initialization for smooth updates
        self.signal_buffer_size = int(self.fs * self.plot_duration)
        self.bvp_buffer = collections.deque(maxlen=self.signal_buffer_size)
        self.rsp_buffer = collections.deque(maxlen=self.signal_buffer_size)


    def init_buffers(self):
        self.inference_frame_buffer = np.zeros(
            (1, self.in_channels, self.num_frames, self.height, self.width))
        self.estimated_bvp = np.array([])
        self.estimated_rsp = np.array([])
        self.hr_list = []
        self.rr_list = []

    def inference_thread(self):
        count_frame = 0
        measurements_recorded = 0

        print("Starting inference thread...")  # Debug print

        while not self.stop_event.is_set():
            data = self.frame_queue.get()
            if data is None:
                break

            frame_idx, face_frame, processed_frame, face_detected = data

            if face_detected and processed_frame is not None:
                if count_frame < self.num_frames:
                    self.inference_frame_buffer[:, :, count_frame, :, :] = processed_frame
                else:
                    self.inference_frame_buffer[:, :, :-1, :, :] = self.inference_frame_buffer[:, :, 1:, :, :]
                    self.inference_frame_buffer[:, :, -1, :, :] = processed_frame

                if count_frame >= self.num_frames and count_frame % self.inference_interval == 0:
                    bvp, rsp = self.model_instance.infer_rphys(self.inference_frame_buffer)
                    bvp, rsp = self.signal_processor.post_process(bvp, rsp)

                    self.bvp_buffer.extend(bvp.reshape(-1))
                    self.rsp_buffer.extend(rsp.reshape(-1))

                    hr, rr = self.signal_processor.compute_metrics(
                        np.array(list(self.bvp_buffer)),
                        np.array(list(self.rsp_buffer))
                    )

                    if not np.isnan(hr) and not np.isnan(rr):
                        measurements_recorded += 1
                        if measurements_recorded % 10 == 0:  # Print every 10 measurements
                            print(f"Recorded {measurements_recorded} valid measurements. HR: {hr:.1f}, RR: {rr:.1f}")

                    self.result_queue.put((face_frame, np.array(list(self.bvp_buffer)),
                                        np.array(list(self.rsp_buffer)), hr, rr, face_detected))
            else:
                self.result_queue.put((None, np.array(list(self.bvp_buffer)) if self.bvp_buffer else np.zeros(self.signal_buffer_size),
                                    np.array(list(self.rsp_buffer)) if self.rsp_buffer else np.zeros(self.signal_buffer_size),
                                    np.nan, np.nan, False))

            count_frame += 1

        print(f"Inference thread ended. Recorded {measurements_recorded} valid measurements")  # Debug print
        self.result_queue.put(None)
