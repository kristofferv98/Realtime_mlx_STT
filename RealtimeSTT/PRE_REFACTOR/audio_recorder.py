
"""

The AudioToTextRecorder class in the provided code facilitates
fast speech-to-text transcription using an MLX-optimized backend.

The class employs the MLX-optimized Whisper model to transcribe recorded audio
into text. Voice activity detection (VAD) is built in, meaning the software can
automatically start or stop recording based on the presence or absence of
speech. It integrates wake word detection, allowing the software to initiate
recording when a specific word or phrase is spoken. The system provides
real-time feedback and can be further customized.

Features:
- Voice Activity Detection: Automatically starts/stops recording when speech
  is detected or when speech ends.
- Wake Word Detection: Starts recording when a specified wake word (or words)
  is detected.
- Event Callbacks: Customizable callbacks for when recording starts
  or finishes.
- Fast Transcription: Returns the transcribed text from the audio as fast
  as possible, leveraging Apple Silicon optimizations via MLX.

Author: Kolja Beigel (Original version)
        Kristoffer Vatnehol (MLX Implementation)

"""
from typing import Iterable, List, Optional, Union
from ctypes import c_bool
from scipy import signal # Import signal for resampling functions
from .mlx_transcriber import MLXTranscriber
import soundfile as sf # Keep for warmup audio if MLXTranscriber needs it
import openwakeword
from openwakeword.model import Model
import numpy as np
import traceback
import threading
import webrtcvad
import datetime
import platform
import logging
import struct
import base64 # If last_transcription_bytes_b64 is needed
import queue # Keep for audio_queue
import torch # Keep for Silero VAD
import halo
import collections # Keep for audio_buffer, last_words_buffer
import pvporcupine # Keep for wake word
import time
import copy
import os
import gc
import torch.multiprocessing as mp # Keep for _audio_data_worker if using mp.Process

# Named logger for this module.
logger = logging.getLogger("realtimestt")
logger.propagate = False

# Set OpenMP runtime duplicate library handling to OK (Use only for development!)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Constants from the original file, adjust as necessary for MLX
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_REALTIME_INITIAL_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 100

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000 # MLX Whisper typically expects 16kHz
BUFFER_SIZE = 512 # This is for PyAudio, Silero/MLX might have different needs
INT16_MAX_ABS_VALUE = 32768.0

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True


class bcolors:
    OKGREEN = '\033[92m'  # Green for active speech detection
    WARNING = '\033[93m'  # Yellow for silence detection
    ENDC = '\033[0m'      # Reset to default color


class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    MLX-optimized Whisper large-v3-turbo model.
    
    This class leverages Apple Silicon's performance through MLX for high-speed, 
    accurate transcription. It supports both batch and streaming modes, with 
    built-in voice activity detection and optional wake word recognition.
    
    The MLX backend provides significant speed improvements on Apple Silicon
    compared to traditional CPU/GPU implementations.
    """

    def __init__(self,
                 # MLX-specific parameters
                 mlx_model_path: str = "openai/whisper-large-v3-turbo",
                 mlx_quick_mode: bool = True,
                 
                 # General parameters
                 language: str = "",
                 input_device_index: int = None,
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 level=logging.WARNING,

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 init_realtime_after_seconds=INIT_REALTIME_INITIAL_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,

                 # Voice activation parameters
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = INIT_POST_SPEECH_SILENCE_DURATION,
                 min_length_of_recording: float = INIT_MIN_LENGTH_OF_RECORDING,
                 min_gap_between_recordings: float = INIT_MIN_GAP_BETWEEN_RECORDINGS,
                 pre_recording_buffer_duration: float = INIT_PRE_RECORDING_BUFFER_DURATION,
                 on_vad_start=None,
                 on_vad_stop=None,
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,
                 on_turn_detection_start=None,
                 on_turn_detection_stop=None,
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,

                 # Wake word parameters
                 wakeword_backend: str = "",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = INIT_WAKE_WORD_ACTIVATION_DELAY,
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 
                 # Other parameters
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logging: bool = False,
                 normalize_audio: bool = False,
                 start_callback_in_new_thread: bool = False,
                 ):

        self.language = language
        self.input_device_index = input_device_index
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = ensure_sentence_starting_uppercase
        self.ensure_sentence_ends_with_period = ensure_sentence_ends_with_period
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_turn_detection_start = on_turn_detection_start
        self.on_turn_detection_stop = on_turn_detection_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.mlx_model_path = mlx_model_path
        self.mlx_quick_mode = mlx_quick_mode
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_processing_pause = realtime_processing_pause
        self.init_realtime_after_seconds = init_realtime_after_seconds
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.on_realtime_transcription_stabilized = on_realtime_transcription_stabilized
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.allowed_latency_limit = allowed_latency_limit

        self.level = level
        self.audio_queue = mp.Queue() # For _audio_data_worker
        self.buffer_size = buffer_size # For PyAudio
        self.sample_rate = sample_rate # For PyAudio & VAD
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_deactivity_detection = silero_deactivity_detection
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = [] # For realtime stabilization
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        # self.audio_interface = None # Managed by _audio_data_worker or AudioInput class
        self.audio = None # This will store the final audio as numpy array
        self.stream = None # PyAudio stream, managed by _audio_data_worker
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.last_transcription_bytes = None
        # self.last_transcription_bytes_b64 = None # If needed
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}
        self.detected_language = None # MLXTranscriber might not expose this directly
        self.detected_language_probability = 0
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.shutdown_lock = threading.Lock()
        # self.transcribe_count = 0 # Not needed with MLXTranscriber's queue
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.use_extended_logging = use_extended_logging
        self.normalize_audio = normalize_audio
        self.awaiting_speech_end = False
        self.start_callback_in_new_thread = start_callback_in_new_thread

        # Logger Configuration
        logger.setLevel(logging.DEBUG)
        log_format = "RealTimeSTT: %(name)s - %(levelname)s - %(message)s"
        file_log_format = "%(asctime)s.%(msecs)03d - " + log_format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        if not no_log_file:
            file_handler = logging.FileHandler('realtimesst_mlx.log') # Changed log file name
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(file_log_format, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(file_handler)

        self.is_shut_down = False
        self.shutdown_event = mp.Event() # For _audio_data_worker
        
        try:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logger.info(f"Start method has already been set. Details: {e}")

        logger.info("Starting RealTimeSTT with MLX Backend")

        if use_extended_logging:
            logger.info("RealtimeSTT_MLX was called with these parameters:")
            # Be careful logging 'self' directly, log specific relevant params
            # for param, value in locals().items(): 
            # if param != 'self': logger.info(f"{param}: {value}")
            logger.info(f"mlx_model_path: {mlx_model_path}, language: {language}, enable_realtime_transcription: {enable_realtime_transcription}")


        self.interrupt_stop_event = mp.Event() # For _audio_data_worker and main loop
        self.was_interrupted = mp.Event()

        # Initialize MLX Transcriber
        if not (platform.system() == 'Darwin' and platform.machine() == 'arm64'):
            raise RuntimeError("Realtime_mlx_STT (MLX backend) requires Apple Silicon (macOS arm64)")

        self.mlx_transcriber = MLXTranscriber(
            model_path=self.mlx_model_path,
            realtime_mode=self.enable_realtime_transcription,
            any_lang=not self.language,
            quick=self.mlx_quick_mode,
            language=self.language if self.language else None
        )
        self.mlx_transcriber.start()
        logger.info(f"MLX transcriber initialized and started with model: {self.mlx_model_path}")

        # Start audio data reading process if using microphone
        if self.use_microphone.value:
            logger.info("Initializing audio recording (creating PyAudio input stream)")
            self.reader_process = self._start_thread(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size, # PyAudio buffer size
                    self.input_device_index,
                    self.shutdown_event, # mp.Event for process
                    self.interrupt_stop_event, # mp.Event for process
                    self.use_microphone # mp.Value
                )
            )

        # Setup wake word detection (remains largely the same)
        if wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords', 'pvp', 'pvporcupine'}:
            self.wakeword_backend = wakeword_backend
            self.wake_words_list = [word.strip() for word in wake_words.lower().split(',')]
            self.wake_words_sensitivities = [float(wake_words_sensitivity) for _ in range(len(self.wake_words_list))]

            if wake_words and self.wakeword_backend in {'pvp', 'pvporcupine'}:
                try:
                    self.porcupine = pvporcupine.create(keywords=self.wake_words_list, sensitivities=self.wake_words_sensitivities)
                    # self.buffer_size = self.porcupine.frame_length # PyAudio buffer_size is different
                    # self.sample_rate = self.porcupine.sample_rate # Should match global SAMPLE_RATE
                    if self.porcupine.sample_rate != SAMPLE_RATE:
                        logger.warning(f"Porcupine sample rate {self.porcupine.sample_rate} differs from desired {SAMPLE_RATE}. Ensure audio passed to Porcupine is at its expected rate.")
                except Exception as e:
                    logger.exception(f"Error initializing porcupine: {e}. Wakewords: {self.wake_words_list}.")
                    raise
                logger.debug("Porcupine wake word detection engine initialized successfully")
            elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}: # Allow oww without specific wake_words
                openwakeword.utils.download_models()
                try:
                    model_paths_list = openwakeword_model_paths.split(',') if openwakeword_model_paths else None
                    self.owwModel = Model(wakeword_models=model_paths_list, inference_framework=openwakeword_inference_framework)
                    self.oww_n_models = len(self.owwModel.models.keys())
                    if not self.oww_n_models: logger.error("No openwakeword models loaded.")
                    for model_key in self.owwModel.models.keys(): logger.info(f"Successfully loaded openwakeword model: {model_key}")
                except Exception as e:
                    logger.exception(f"Error initializing openwakeword: {e}")
                    raise
                logger.debug("Open wake word detection engine initialized successfully")
            elif self.use_wake_words: # If use_wake_words was true but backend invalid
                 logger.error(f"Wakeword engine {self.wakeword_backend} unknown/unsupported or wake_words not specified. Please specify one of: pvporcupine, openwakeword.")


        # Setup voice activity detection model WebRTC (remains the same)
        try:
            logger.info(f"Initializing WebRTC voice with Sensitivity {webrtc_sensitivity}")
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)
        except Exception as e:
            logger.exception(f"Error initializing WebRTC VAD: {e}")
            raise
        logger.debug("WebRTC VAD engine initialized successfully")

        # Setup voice activity detection model Silero VAD (remains the same)
        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                verbose=False, onnx=silero_use_onnx
            )
        except Exception as e:
            logger.exception(f"Error initializing Silero VAD: {e}")
            raise
        logger.debug("Silero VAD engine initialized successfully")

        self.audio_buffer = collections.deque(maxlen=int((self.sample_rate // self.buffer_size) * self.pre_recording_buffer_duration))
        self.last_words_buffer = collections.deque(maxlen=int((self.sample_rate // self.buffer_size) * 0.3)) # For VAD context
        self.frames = [] # Stores bytes audio chunks for current recording
        self.last_frames = [] # Stores bytes audio chunks from last recording for context

        self.is_recording = False
        self.is_running = True # Controls the main loops
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
                   
        # No separate stdout_thread needed as MLXTranscriber logs via Python logging
        logger.debug('RealtimeSTT_MLX initialization completed successfully')
                   
    def _start_thread(self, target=None, args=()):
        # Using mp.Process for _audio_data_worker as it involves PyAudio which can be tricky with threads
        # Other internal threads (recording_thread, realtime_thread) are standard threading.Thread
        if target == AudioToTextRecorder._audio_data_worker:
             # Ensure spawn method for PyAudio compatibility on macOS/Windows if _audio_data_worker is a process
            # if mp.get_start_method(allow_none=True) is None and (platform.system() != 'Linux'):
            #     mp.set_start_method("spawn", force=True) # force=True if already set by other means
            # elif platform.system() == 'Linux': # Linux can use fork
            #     if mp.get_start_method(allow_none=True) is None: mp.set_start_method("fork", force=True)

            thread = mp.Process(target=target, args=args)
            thread.start() # mp.Process uses start()
            return thread
        else: # For _recording_worker, _realtime_worker
            thread = threading.Thread(target=target, args=args)
            thread.daemon = True # Ensure they exit when main program exits
            thread.start()
            return thread

    def _read_stdout(self):
        # This method is no longer needed as MLXTranscriber uses Python's logging directly.
        pass

    def _run_callback(self, cb, *args, **kwargs):
        if self.start_callback_in_new_thread:
            threading.Thread(target=cb, args=args, kwargs=kwargs, daemon=True).start()
        else:
            cb(*args, **kwargs)

    @staticmethod
    def _audio_data_worker(
        audio_queue, # mp.Queue
        target_sample_rate,
        buffer_size, # PyAudio buffer_size
        input_device_index,
        shutdown_event, # mp.Event
        interrupt_stop_event, # mp.Event
        use_microphone # mp.Value
    ):
        # This static method remains largely the same as it deals with PyAudio directly.
        # It should correctly put byte chunks into audio_queue.
        import pyaudio # Import locally for the process
        # numpy and scipy.signal also imported locally for the process
        import numpy as np_process 
        from scipy import signal as signal_process

        # ... (rest of _audio_data_worker implementation, ensure it uses np_process and signal_process) ...
        # Ensure it uses target_sample_rate (which is self.sample_rate from __init__)
        # and buffer_size (which is self.buffer_size for PyAudio from __init__)
        # The key is that it outputs raw audio bytes to audio_queue.
        
        # Example of a relevant part:
        # def preprocess_audio(chunk, original_sample_rate, target_sample_rate_local):
        # ...
        # if original_sample_rate != target_sample_rate_local:
        # num_samples = int(len(chunk) * target_sample_rate_local / original_sample_rate)
        # chunk = signal_process.resample(chunk, num_samples)
        # chunk = chunk.astype(np_process.int16)
        # ...
        # return chunk.tobytes()

        # Inside the loop:
        # processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
        # audio_queue.put(to_process) # This puts bytes into the mp.Queue

        # (Full _audio_data_worker code from previous diff should be here,
        # just ensure local imports and correct variable usage for sample_rate and buffer_size)
        # For brevity, I'm not repeating the entire _audio_data_worker here.
        # Assume it's the same as the one provided in your initial files.
        # Key: it puts raw audio byte chunks into `audio_queue`.
        # Ensure it uses `target_sample_rate` which is `self.sample_rate` (e.g., 16000)
        # And `buffer_size` which is `self.buffer_size` for PyAudio (e.g., 512)
        # The Silero buffer size logic inside this worker needs to be adapted or removed
        # if Silero processing is now done in _recording_worker based on self.buffer_size (audio frames).

        # --- BEGINNING of _audio_data_worker (from original, ensure local imports if needed) ---
        # Note: This is a placeholder. The full, correct _audio_data_worker logic
        # from your original file, adapted for local imports if it's a separate process,
        # needs to be here. The critical part is that it reads from PyAudio and puts
        # byte chunks into `audio_queue`.

        # Simplified for this example:
        # import pyaudio
        # import numpy as np_process # Renamed to avoid conflict if this were in the same file
        # from scipy import signal as signal_process

        # audio_interface_local = pyaudio.PyAudio()
        # stream_local = audio_interface_local.open(
        #     format=pyaudio.paInt16, channels=1, rate=target_sample_rate, input=True,
        #     frames_per_buffer=buffer_size, input_device_index=input_device_index
        # )
        # silero_buffer_size_vad = 2 * buffer_size # For VAD processing needs typically 30ms chunks (512 at 16kHz is ~32ms)
        # internal_buffer = bytearray()

        # try:
        #     while not shutdown_event.is_set():
        #         if not use_microphone.value:
        #             time.sleep(0.01)
        #             continue
        #         try:
        #             data = stream_local.read(buffer_size, exception_on_overflow=False)
        #             internal_buffer += data
        #             while len(internal_buffer) >= silero_buffer_size_vad:
        #                 to_process = internal_buffer[:silero_buffer_size_vad]
        #                 internal_buffer = internal_buffer[silero_buffer_size_vad:]
        #                 audio_queue.put(to_process)
        #         except OSError as e_os:
        #             if e_os.errno == pyaudio.paInputOverflowed: logger.warning("PyAudio Input Overflowed")
        #             else: raise
        #         except KeyboardInterrupt: interrupt_stop_event.set(); break
        # except Exception as e_main:
        #     logger.error(f"_audio_data_worker error: {e_main}", exc_info=True)
        # finally:
        #     if 'stream_local' in locals() and stream_local: stream_local.close()
        #     if 'audio_interface_local' in locals() and audio_interface_local: audio_interface_local.terminate()
        # --- END of simplified _audio_data_worker ---
        # The full _audio_data_worker from the original `audio_recorder.py` should be used,
        # ensuring it uses `target_sample_rate` (which is `self.sample_rate`) and `buffer_size` (for PyAudio)
        # and correctly puts byte chunks into `audio_queue`.
        # The key is that audio_queue gets bytes.

        # --- Full _audio_data_worker from your provided code (with minor adjustments for clarity) ---
        import pyaudio as pa_local # Local import for process safety
        import numpy as np_local
        from scipy import signal as sig_local

        audio_interface = None
        stream = None
        # device_sample_rate is the actual rate the device is opened at
        # target_sample_rate is what VAD/STT expects (e.g. 16000)
        
        def get_highest_sample_rate_local(audio_interface_local_fn, device_index_local_fn):
            try:
                device_info = audio_interface_local_fn.get_device_info_by_index(device_index_local_fn)
                # Simplified: assume default is good enough or try target_sample_rate directly
                return int(device_info['defaultSampleRate'])
            except: return target_sample_rate # Fallback

        def initialize_audio_stream_local(audio_interface_local_fn, sample_rate_local_fn, chunk_size_local_fn, input_dev_idx_local_fn):
            # Simplified validation
            try:
                stream_local_fn = audio_interface_local_fn.open(
                    format=pa_local.paInt16, channels=1, rate=sample_rate_local_fn, input=True,
                    frames_per_buffer=chunk_size_local_fn, input_device_index=input_dev_idx_local_fn,
                )
                return stream_local_fn, sample_rate_local_fn
            except Exception as e_init_stream:
                logger.warning(f"Could not open stream at {sample_rate_local_fn}Hz for device {input_dev_idx_local_fn}: {e_init_stream}")
                # Try with default device info if specific index failed or was None
                if input_dev_idx_local_fn is not None: # If a specific device failed, try default
                    try:
                        default_device_info = audio_interface_local_fn.get_default_input_device_info()
                        input_dev_idx_local_fn = default_device_info['index']
                        sample_rate_local_fn = int(default_device_info['defaultSampleRate'])
                        stream_local_fn = audio_interface_local_fn.open(
                            format=pa_local.paInt16, channels=1, rate=sample_rate_local_fn, input=True,
                            frames_per_buffer=chunk_size_local_fn, input_device_index=input_dev_idx_local_fn,
                        )
                        logger.info(f"Successfully opened default input device {input_dev_idx_local_fn} at {sample_rate_local_fn}Hz")
                        return stream_local_fn, sample_rate_local_fn
                    except Exception as e_default_stream:
                        logger.error(f"Could not open default input device: {e_default_stream}")
                        raise # Re-raise if default also fails
                raise # Re-raise if initial attempt failed and no default was tried

        def preprocess_audio_local(chunk_bytes, original_sr, target_sr):
            if not isinstance(chunk_bytes, bytes): return chunk_bytes # Should already be bytes
            
            audio_np = np_local.frombuffer(chunk_bytes, dtype=np_local.int16)
            if original_sr != target_sr:
                num_samples_target = int(len(audio_np) * target_sr / original_sr)
                audio_np = sig_local.resample(audio_np, num_samples_target)
                audio_np = audio_np.astype(np_local.int16)
            return audio_np.tobytes()

        try:
            audio_interface = pa_local.PyAudio()
            actual_input_device_index = input_device_index
            if actual_input_device_index is None:
                actual_input_device_index = audio_interface.get_default_input_device_info()['index']

            # Attempt to open stream with target_sample_rate first, then device's default if that fails
            opened_rate = target_sample_rate # What VAD/STT expect
            try:
                stream, device_opened_at_rate = initialize_audio_stream_local(audio_interface, opened_rate, buffer_size, actual_input_device_index)
            except Exception: # If target_sample_rate fails, try device's default
                logger.warning(f"Failed to open stream at {opened_rate}Hz, trying device default.")
                device_default_rate = get_highest_sample_rate_local(audio_interface, actual_input_device_index)
                stream, device_opened_at_rate = initialize_audio_stream_local(audio_interface, device_default_rate, buffer_size, actual_input_device_index)
            
            logger.info(f"Audio stream opened at {device_opened_at_rate}Hz for device {actual_input_device_index}. Target for VAD/STT is {target_sample_rate}Hz.")

            # Silero VAD expects chunks of specific sizes (e.g., 256, 512, 768, 1024, 1536 samples for 16kHz)
            # A common VAD chunk is ~30ms. For 16kHz, 30ms = 480 samples. BUFFER_SIZE=512 is close.
            # The queue will receive chunks of `vad_chunk_size_bytes` after potential resampling.
            vad_chunk_duration_ms = 30 # Standard VAD chunk size
            vad_chunk_num_samples = int(target_sample_rate * (vad_chunk_duration_ms / 1000.0))
            vad_chunk_size_bytes = vad_chunk_num_samples * 2 # 2 bytes per int16 sample

            internal_byte_buffer = bytearray()

            while not shutdown_event.is_set():
                if not use_microphone.value:
                    time.sleep(0.01)
                    continue
                try:
                    # Read raw data from PyAudio stream
                    raw_data_chunk = stream.read(buffer_size, exception_on_overflow=False)
                    
                    # Preprocess: primarily resample if device_opened_at_rate is different from target_sample_rate
                    processed_data_chunk = preprocess_audio_local(raw_data_chunk, device_opened_at_rate, target_sample_rate)
                    
                    internal_byte_buffer.extend(processed_data_chunk)

                    while len(internal_byte_buffer) >= vad_chunk_size_bytes:
                        chunk_to_queue = internal_byte_buffer[:vad_chunk_size_bytes]
                        internal_byte_buffer = internal_byte_buffer[vad_chunk_size_bytes:]
                        audio_queue.put(chunk_to_queue)

                except OSError as e_os:
                    if e_os.errno == pa_local.paInputOverflowed: logger.warning("PyAudio Input Overflowed in _audio_data_worker.")
                    else: 
                        logger.error(f"OSError in _audio_data_worker: {e_os}", exc_info=True)
                        # Attempt to reinitialize
                        try:
                            if stream: stream.close()
                            if audio_interface: audio_interface.terminate()
                            audio_interface = pa_local.PyAudio()
                            stream, device_opened_at_rate = initialize_audio_stream_local(audio_interface, target_sample_rate, buffer_size, actual_input_device_index)
                        except Exception as e_reinit:
                            logger.error(f"Failed to reinitialize audio stream: {e_reinit}. Exiting worker.")
                            break # Exit loop
                    continue # Continue to next read attempt
                except KeyboardInterrupt: 
                    interrupt_stop_event.set()
                    break
                except Exception as e_loop:
                    logger.error(f"Error in _audio_data_worker loop: {e_loop}", exc_info=True)
                    # Potentially add a small sleep to prevent rapid error loops
                    time.sleep(0.1)


        except Exception as e_main_worker:
            logger.error(f"Critical error in _audio_data_worker setup: {e_main_worker}", exc_info=True)
        finally:
            if stream: stream.close()
            if audio_interface: audio_interface.terminate()
            logger.info("_audio_data_worker finished.")
        # --- END of Full _audio_data_worker ---


    def wakeup(self):
        self.listen_start = time.time()

    def abort(self):
        # Simpler abort for MLX: just stop recording and don't transcribe current buffer.
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.interrupt_stop_event.set() # Signal recording worker
        if self.is_recording:
            self.stop() # This will set is_recording to False
        self.frames.clear() # Clear any buffered frames
        self.audio_buffer.clear()
        self.was_interrupted.set() # Notify if text() was waiting
        self._set_state("inactive")

    def wait_audio(self):
        try:
            logger.info("Setting listen time")
            if self.listen_start == 0:
                self.listen_start = time.time()

            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True
                logger.debug('Waiting for recording start')
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02): break
                if self.interrupt_stop_event.is_set(): return # Interrupted

            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True
                logger.debug('Waiting for recording stop')
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout=0.02)): break
                if self.interrupt_stop_event.is_set(): return # Interrupted

            # --- Consolidate frames into self.audio (numpy array) ---
            # self.frames now contains bytes chunks from _recording_worker
            if not self.frames and self.last_frames: # If current recording empty, use last
                logger.info("No frames in current recording, using last_frames for transcription.")
                concatenated_audio_bytes = b''.join(self.last_frames)
            else:
                concatenated_audio_bytes = b''.join(self.frames)

            if not concatenated_audio_bytes:
                logger.info("No audio frames to process.")
                self.audio = np.array([], dtype=np.float32)
            else:
                full_audio_array_int16 = np.frombuffer(concatenated_audio_bytes, dtype=np.int16)
                self.audio = full_audio_array_int16.astype(np.float32) / INT16_MAX_ABS_VALUE
            
            # Handle backdate_resume_seconds: what to keep in self.frames for next potential recording
            frames_to_keep_for_resume = []
            if self.backdate_resume_seconds > 0 and self.audio.size > 0:
                samples_to_keep_for_resume = int(self.sample_rate * self.backdate_resume_seconds)
                samples_to_keep_for_resume = min(samples_to_keep_for_resume, len(self.audio))
                
                # Get the tail of the self.audio that corresponds to backdate_resume_seconds
                audio_to_keep_for_resume_np = self.audio[-samples_to_keep_for_resume:]
                audio_to_keep_for_resume_int16 = (audio_to_keep_for_resume_np * INT16_MAX_ABS_VALUE).astype(np.int16)
                bytes_to_keep_for_resume = audio_to_keep_for_resume_int16.tobytes()
                
                # Split into VAD-sized chunks (e.g., 512 samples * 2 bytes/sample = 1024 bytes)
                # This assumes self.buffer_size is used by VAD in _recording_worker
                # The audio in self.frames are already in chunks (typically from audio_queue)
                # A more robust way might be to reconstruct based on number of original frames.
                # For simplicity, let's assume vad_chunk_size_bytes is known or can be derived.
                vad_chunk_duration_ms = 30 
                vad_chunk_num_samples = int(self.sample_rate * (vad_chunk_duration_ms / 1000.0))
                vad_chunk_size_bytes = vad_chunk_num_samples * 2

                for i in range(0, len(bytes_to_keep_for_resume), vad_chunk_size_bytes):
                    frame_chunk = bytes_to_keep_for_resume[i : i + vad_chunk_size_bytes]
                    if frame_chunk: frames_to_keep_for_resume.append(frame_chunk)
            
            self.frames.clear()
            self.frames.extend(frames_to_keep_for_resume)
            self.last_frames.clear() # Clear any cached frames from previous transcription

            # Handle backdate_stop_seconds: what to remove from self.audio for transcription
            if self.backdate_stop_seconds > 0 and self.audio.size > 0:
                samples_to_remove_from_transcription = int(self.sample_rate * self.backdate_stop_seconds)
                if samples_to_remove_from_transcription < len(self.audio):
                    self.audio = self.audio[:-samples_to_remove_from_transcription]
                else:
                    self.audio = np.array([], dtype=np.float32) # Remove all if backdate is too long
            
            self.backdate_stop_seconds = 0.0
            self.backdate_resume_seconds = 0.0
            self.listen_start = 0
            self._set_state("inactive")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise

    def perform_final_transcription(self, audio_bytes_np=None, use_prompt=True): # use_prompt might be less relevant for MLX
        # MLX backend is now the only path
        transcription_start_time_mlx = time.time()
        
        current_audio_to_transcribe_np = audio_bytes_np
        if current_audio_to_transcribe_np is None: 
            current_audio_to_transcribe_np = copy.deepcopy(self.audio) # self.audio is already np.float32

        if current_audio_to_transcribe_np is None or len(current_audio_to_transcribe_np) == 0:
            logger.info("No audio data available for MLX transcription")
            self._set_state("inactive") # Ensure state is reset
            return ""

        # Normalization: MLXTranscriber/whisper_turbo.py handles normalization internally.
        # If self.normalize_audio is True, it implies a custom normalization step,
        # but the MLX pipeline might do its own. For now, pass as is.
        # if self.normalize_audio:
        #    current_audio_to_transcribe_np = self._normalize_audio(current_audio_to_transcribe_np)

        self.mlx_transcriber.transcribe(current_audio_to_transcribe_np) 

        result = None
        max_wait = 60.0 # seconds
        wait_start = time.time()

        while time.time() - wait_start < max_wait:
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
                self._set_state("inactive")
                return ""
            
            result = self.mlx_transcriber.get_result(timeout=0.5)
            if result and result.get('is_final', True): 
                break
        
        self._set_state("inactive") 
        if result:
            transcription = result['text']
            processing_time = result['processing_time']
            self.last_transcription_bytes = copy.deepcopy(current_audio_to_transcribe_np)
            # self.last_transcription_bytes_b64 = base64.b64encode(self.last_transcription_bytes.tobytes()).decode('utf-8')
            transcription = self._preprocess_output(transcription)
            
            if self.print_transcription_time:
                total_time = time.time() - transcription_start_time_mlx
                logger.info(f"MLX transcription in {processing_time:.2f}s (total: {total_time:.2f}s)")
            return "" if self.interrupt_stop_event.is_set() else transcription
        else:
            logger.error("MLX transcription timed out or failed")
            return ""

    def transcribe(self):
        """
        Transcribe the recorded audio using the MLX-optimized Whisper model.
        
        This method processes the audio data stored in self.audio through the MLX
        transcription pipeline. It first makes a copy of the audio to prevent
        modifications to the original data.
        
        If an on_transcription_start callback is defined, it will be called with
        the audio data before the actual transcription begins. The callback can:
        1. Return False to explicitly abort the transcription process
        2. Return anything else (True, None, etc.) to proceed with transcription
        3. Modify the audio data (though this isn't recommended)
        
        Returns:
            str: The transcribed text, or an empty string if transcription was aborted
                 or if no audio was available.
        """
        # self.audio should be a float32 numpy array from wait_audio()
        audio_to_process_np = copy.deepcopy(self.audio) 
        self._set_state("transcribing")
        
        # If on_transcription_start callback is defined, call it
        if self.on_transcription_start:
            # The callback receives the audio and can decide whether to abort
            # It might also modify audio_to_process_np (allowed but not expected)
            # Return value semantics:
            #   - If the callback returns False (explicitly), abort transcription
            #   - Any other return value (True, None, etc.) means proceed
            should_proceed = self.on_transcription_start(audio_to_process_np)
            
            # Only abort if callback explicitly returns False
            if should_proceed is False:
                logger.info("Transcription aborted by on_transcription_start callback.")
                self._set_state("inactive")
                return ""  # Return empty string for aborted transcription
        
        # Proceed with transcription (in all cases except explicit abort)
        return self.perform_final_transcription(audio_to_process_np)

    def _process_wakeword(self, data_bytes): # Expects bytes
        if self.wakeword_backend in {'pvp', 'pvporcupine'}:
            # Porcupine expects int16 PCM data
            try:
                pcm = struct.unpack_from("h" * (len(data_bytes) // 2), data_bytes)
                porcupine_index = self.porcupine.process(pcm)
                if self.debug_mode: logger.info(f"Wake words porcupine_index: {porcupine_index}")
                return porcupine_index
            except struct.error:
                 logger.error("Error unpacking audio data for Porcupine wake word processing.", exc_info=True)
                 return -1
            except Exception as e_pwp: # Catch specific Porcupine errors if any
                 logger.error(f"Porcupine processing error: {e_pwp}", exc_info=True)
                 return -1

        elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            # OpenWakeWord expects int16 numpy array
            try:
                pcm_np = np.frombuffer(data_bytes, dtype=np.int16)
                prediction = self.owwModel.predict(pcm_np) # owwModel expects numpy array
                max_score = -1
                max_index = -1
                if self.owwModel.prediction_buffer: # Check if buffer has keys
                    for idx, mdl_key in enumerate(self.owwModel.prediction_buffer.keys()):
                        scores = list(self.owwModel.prediction_buffer[mdl_key])
                        if scores and scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
                            max_score = scores[-1]
                            max_index = idx # This idx might not map directly if wake_words_list isn't used by oww
                if self.debug_mode: logger.info(f"Wake words oww max_index: {max_index}, max_score: {max_score}")
                return max_index 
            except Exception as e_oww:
                logger.error(f"OpenWakeWord processing error: {e_oww}", exc_info=True)
                return -1
        
        if self.debug_mode: logger.info("Wake words no match or backend not supported/configured.")
        return -1


    def text(self, on_transcription_finished=None):
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in text() method")
            self.shutdown()
            raise 
        
        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        transcribed_text = self.transcribe() # transcribe now returns the text or None
        if transcribed_text is None: transcribed_text = "" # Ensure string return

        if on_transcription_finished:
            # Run callback in a new thread if desired by user, otherwise synchronously
            if self.start_callback_in_new_thread:
                 threading.Thread(target=on_transcription_finished, args=(transcribed_text,)).start()
            else:
                 on_transcription_finished(transcribed_text)
            return None # Consistent with original if callback is used
        else:
            return transcribed_text


    def format_number(self, num): # Utility, keep as is
        num_str = f"{num:.10f}"
        integer_part, decimal_part = num_str.split('.')
        result = f"{integer_part[-2:]}.{decimal_part[:2]}"
        return result

    def start(self, frames_bytes_initial = None): # Expects optional initial byte frames
        if (time.time() - self.recording_stop_time < self.min_gap_between_recordings):
            logger.info("Attempted to start recording too soon after stopping.")
            return self
        
        logger.info("Recording started")
        self._set_state("recording")
        self.text_storage.clear() # Clear for new realtime session
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames.clear() # self.frames stores bytes chunks
        if frames_bytes_initial: # If initial frames (bytes) are provided
            if isinstance(frames_bytes_initial, list) and all(isinstance(f, bytes) for f in frames_bytes_initial):
                self.frames.extend(frames_bytes_initial)
            elif isinstance(frames_bytes_initial, bytes):
                 # If a single bytes object, split it into VAD-sized chunks if needed or store as one
                 # For simplicity, assume it's already chunked or can be appended directly if _recording_worker expects that.
                 # If _recording_worker processes self.frames directly, it needs to handle potentially large initial byte string.
                 # Better to ensure frames_bytes_initial is a list of VAD-sized byte chunks.
                 # This part might need refinement based on how initial frames are fed.
                 # For now, let's assume it's a list of appropriate byte chunks.
                 logger.warning("Starting with raw bytes directly in self.frames. Ensure _recording_worker can handle this if not VAD-chunked.")
                 self.frames.append(frames_bytes_initial) # Or extend if it's a list of chunks
            else:
                 logger.error("Invalid format for frames_bytes_initial. Expected list of bytes or bytes.")

        self.is_recording = True
        self.recording_start_time = time.time()
        self.is_silero_speech_active = False # Reset VAD states
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self._run_callback(self.on_recording_start)
        return self

    def stop(self, backdate_stop_seconds: float = 0.0, backdate_resume_seconds: float = 0.0):
        if (time.time() - self.recording_start_time < self.min_length_of_recording):
            logger.info("Attempted to stop recording too soon after starting.")
            return self
        
        logger.info("Recording stopped")
        self.last_frames = copy.deepcopy(self.frames) # Store current frames (bytes) before clearing
        self.backdate_stop_seconds = backdate_stop_seconds
        self.backdate_resume_seconds = backdate_resume_seconds
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.last_recording_start_time = self.recording_start_time
        self.last_recording_stop_time = self.recording_stop_time

        if self.on_recording_stop:
            self._run_callback(self.on_recording_stop)
        return self

    def listen(self): # Keep as is
        self.listen_start = time.time()
        self._set_state("listening")
        self.start_recording_on_voice_activity = True

    def feed_audio(self, chunk_bytes, original_sample_rate=16000): # Expects bytes
        # This method is for non-microphone input. It should put bytes into audio_queue.
        # It needs to resample if original_sample_rate != self.sample_rate (VAD/STT rate)
        # and then chunk it appropriately for VAD.
        if not hasattr(self, 'feed_buffer'): self.feed_buffer = bytearray()
        
        processed_chunk_bytes = chunk_bytes
        if original_sample_rate != self.sample_rate:
            audio_np = np.frombuffer(chunk_bytes, dtype=np.int16)
            num_samples_target = int(len(audio_np) * self.sample_rate / original_sample_rate)
            resampled_np = signal.resample(audio_np, num_samples_target).astype(np.int16)
            processed_chunk_bytes = resampled_np.tobytes()

        self.feed_buffer.extend(processed_chunk_bytes)
        
        vad_chunk_duration_ms = 30 
        vad_chunk_num_samples = int(self.sample_rate * (vad_chunk_duration_ms / 1000.0))
        vad_chunk_size_bytes = vad_chunk_num_samples * 2

        while len(self.feed_buffer) >= vad_chunk_size_bytes:
            to_process = self.feed_buffer[:vad_chunk_size_bytes]
            self.feed_buffer = self.feed_buffer[vad_chunk_size_bytes:]
            self.audio_queue.put(to_process)


    def set_microphone(self, microphone_on=True): # Keep as is
        logger.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on # mp.Value

    def shutdown(self):
        with self.shutdown_lock:
            if self.is_shut_down: return

            logger.info("RealtimeSTT_MLX shutting down") # Changed log message

            self.is_shut_down = True
            self.start_recording_event.set() # Allow loops to exit
            self.stop_recording_event.set()  # Allow loops to exit
            
            self.shutdown_event.set() # For _audio_data_worker (mp.Event)
            self.is_recording = False # Stop _recording_worker loop condition
            self.is_running = False   # Stop _realtime_worker loop condition

            if self.recording_thread and self.recording_thread.is_alive():
                logger.debug('Joining recording thread')
                self.recording_thread.join(timeout=2.0)
                if self.recording_thread.is_alive(): logger.warning("Recording thread did not join.")
            
            if self.realtime_thread and self.realtime_thread.is_alive():
                logger.debug('Joining realtime thread')
                self.realtime_thread.join(timeout=2.0)
                if self.realtime_thread.is_alive(): logger.warning("Realtime thread did not join.")

            if self.use_microphone.value and hasattr(self, 'reader_process') and self.reader_process.is_alive():
                logger.debug('Joining reader process (_audio_data_worker)')
                self.reader_process.join(timeout=5.0) # Increased timeout
                if self.reader_process.is_alive():
                    logger.warning("Reader process (_audio_data_worker) did not terminate. Terminating forcefully.")
                    self.reader_process.terminate()
            
            if self.mlx_transcriber:
                logger.debug("Cleaning up MLX transcriber...")
                self.mlx_transcriber.cleanup()
                self.mlx_transcriber = None
                logger.info("MLX transcriber cleaned up")

            gc.collect()
            logger.info("RealtimeSTT_MLX shutdown complete.")

    def _recording_worker(self):
        # This worker now gets bytes from audio_queue (from _audio_data_worker or feed_audio)
        # It performs VAD, manages self.frames (list of bytes chunks), and calls start/stop.
        if self.use_extended_logging: logger.debug('Debug: Entering _recording_worker try block')
        
        last_inner_try_time = 0
        time_since_last_buffer_message = 0
        was_recording = False # Tracks if we were recording in the previous iteration
        delay_was_passed = False # Tracks if wake_word_activation_delay has passed
        wakeword_detected_time = 0 # Timestamp of last wakeword detection
        wakeword_samples_to_remove_from_frames_bytes = 0 # Bytes to remove after wakeword
        self.allowed_to_early_transcribe = True # Flag for early transcription logic

        try:
            while self.is_running: # is_running is the primary loop control now
                if self.use_extended_logging and last_inner_try_time:
                    processing_duration = time.time() - last_inner_try_time
                    if processing_duration > 0.1: logger.warning(f'### _recording_worker loop took: {processing_duration:.3f}s')
                last_inner_try_time = time.time()

                try:
                    # Get audio data (bytes) from the queue
                    data_bytes = self.audio_queue.get(timeout=0.01) 
                    self.last_words_buffer.append(data_bytes) # For Silero VAD context
                except queue.Empty:
                    if not self.is_running: break
                    # If not recording and no VAD start signal, briefly sleep to avoid busy-waiting if queue is often empty
                    if not self.is_recording and not self.start_recording_on_voice_activity : time.sleep(0.005)
                    continue
                except BrokenPipeError: # Should not happen with mp.Queue
                    logger.error("BrokenPipeError in _recording_worker audio_queue.get()", exc_info=True)
                    self.is_running = False; break
                
                if self.on_recorded_chunk: self._run_callback(self.on_recorded_chunk, data_bytes)

                # Buffer overflow check (on audio_queue size)
                if self.handle_buffer_overflow and self.audio_queue.qsize() > self.allowed_latency_limit:
                    logger.warning(f"Audio queue size ({self.audio_queue.qsize()}) exceeds limit ({self.allowed_latency_limit}). Discarding oldest chunks.")
                    while self.audio_queue.qsize() > self.allowed_latency_limit:
                        try: self.audio_queue.get_nowait()
                        except queue.Empty: break
                
                # --- VAD and Recording Logic ---
                failed_stop_attempt = False # Reset per iteration

                if not self.is_recording:
                    time_since_listen_start = (time.time() - self.listen_start) if self.listen_start else 0
                    current_wake_word_activation_delay_passed = (time_since_listen_start > self.wake_word_activation_delay)

                    if current_wake_word_activation_delay_passed and not delay_was_passed:
                        if self.use_wake_words and self.wake_word_activation_delay and self.on_wakeword_timeout:
                            self._run_callback(self.on_wakeword_timeout)
                    delay_was_passed = current_wake_word_activation_delay_passed
                    
                    # Set UI state (spinner)
                    if not self.recording_stop_time: # if not recently stopped
                        if self.use_wake_words and current_wake_word_activation_delay_passed and not self.wakeword_detected:
                            self._set_state("wakeword")
                        elif self.listen_start : self._set_state("listening")
                        else: self._set_state("inactive")
                    
                    # Wake Word Detection
                    if self.use_wake_words and current_wake_word_activation_delay_passed and not self.wakeword_detected:
                        try:
                            wakeword_index = self._process_wakeword(data_bytes) # Expects bytes
                            if wakeword_index >= 0:
                                self.wake_word_detect_time = time.time()
                                wakeword_detected_time = time.time() # Used for timeout check
                                # Calculate bytes to remove based on wake_word_buffer_duration
                                samples_to_remove_ww = int(self.sample_rate * self.wake_word_buffer_duration)
                                wakeword_samples_to_remove_from_frames_bytes = samples_to_remove_ww * 2 # 2 bytes per int16 sample
                                self.wakeword_detected = True
                                if self.on_wakeword_detected: self._run_callback(self.on_wakeword_detected)
                        except Exception as e_ww: logger.error(f"Wake word processing error: {e_ww}", exc_info=True)
                    
                    # Voice Activity Detection to Start Recording
                    if (self.start_recording_on_voice_activity and (not self.use_wake_words or not current_wake_word_activation_delay_passed or self.wakeword_detected)):
                        if self._is_voice_active(): # Combined VAD check
                            if self.on_vad_start: self._run_callback(self.on_vad_start)
                            logger.info("Voice activity detected, starting recording.")
                            
                            # Collect pre-buffered audio (self.audio_buffer contains byte chunks)
                            initial_frames_from_buffer = list(self.audio_buffer)
                            self.audio_buffer.clear() # Clear pre-buffer after use
                            initial_frames_from_buffer.append(data_bytes) # Add current chunk
                            
                            self.start(frames_bytes_initial=initial_frames_from_buffer) # Pass initial byte frames
                            self.start_recording_on_voice_activity = False
                            self.silero_vad_model.reset_states() # Reset Silero state after starting
                        else: # If no voice active yet, perform VAD checks
                            self._check_voice_activity(data_bytes) # Updates VAD states
                    
                    if self.speech_end_silence_start != 0: # Reset if not recording
                        self.speech_end_silence_start = 0
                        if self.on_turn_detection_stop: self._run_callback(self.on_turn_detection_stop)
                else: # self.is_recording is True
                    # Handle removal of wake word audio from the beginning of self.frames (which stores bytes)
                    if wakeword_samples_to_remove_from_frames_bytes > 0:
                        bytes_removed_count = 0
                        temp_frames = []
                        current_buffer_to_skip = wakeword_samples_to_remove_from_frames_bytes
                        
                        for frame_byte_chunk in self.frames:
                            if current_buffer_to_skip <= 0:
                                temp_frames.append(frame_byte_chunk)
                                continue
                            
                            chunk_len = len(frame_byte_chunk)
                            if current_buffer_to_skip >= chunk_len:
                                current_buffer_to_skip -= chunk_len
                                bytes_removed_count += chunk_len
                            else: # current_buffer_to_skip < chunk_len
                                temp_frames.append(frame_byte_chunk[current_buffer_to_skip:])
                                bytes_removed_count += current_buffer_to_skip
                                current_buffer_to_skip = 0
                        
                        self.frames = temp_frames # Update self.frames
                        if self.debug_mode: logger.info(f"Removed {bytes_removed_count} bytes from start of recording for wake word.")
                        wakeword_samples_to_remove_from_frames_bytes = 0 # Reset flag

                    # Stop Recording on Voice Deactivity
                    if self.stop_recording_on_voice_deactivity:
                        is_speech_now = self._is_silero_speech(data_bytes) if self.silero_deactivity_detection else self._is_webrtc_speech(data_bytes, True)
                        
                        if not is_speech_now:
                            if self.speech_end_silence_start == 0 and (time.time() - self.recording_start_time > self.min_length_of_recording):
                                self.speech_end_silence_start = time.time()
                                self.awaiting_speech_end = True
                                if self.on_turn_detection_start: self._run_callback(self.on_turn_detection_start)
                            
                            # Early Transcription Logic
                            if self.speech_end_silence_start and self.early_transcription_on_silence > 0 and \
                               (time.time() - self.speech_end_silence_start > (self.early_transcription_on_silence / 1000.0)) and \
                               self.allowed_to_early_transcribe and self.frames:
                                logger.debug("Attempting early transcription due to silence.")
                                audio_early_np = np.frombuffer(b''.join(self.frames), dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
                                self.mlx_transcriber.transcribe(audio_early_np) # Request transcription
                                # Result will be picked up by _realtime_worker or perform_final_transcription
                                # No need to increment a transcribe_count here with MLXTranscriber's internal queue.
                                self.allowed_to_early_transcribe = False # Prevent multiple early transcriptions for same silence period
                        else: # Speech is active
                            self.awaiting_speech_end = False
                            if self.speech_end_silence_start != 0:
                                self.speech_end_silence_start = 0
                                if self.on_turn_detection_stop: self._run_callback(self.on_turn_detection_stop)
                            self.allowed_to_early_transcribe = True # Reset if speech resumes

                        # Check for stopping recording
                        if self.speech_end_silence_start and (time.time() - self.speech_end_silence_start >= self.post_speech_silence_duration):
                            if self.on_vad_stop: self._run_callback(self.on_vad_stop)
                            logger.info("Voice deactivity detected, stopping recording.")
                            self.frames.append(data_bytes) # Add current silent chunk before stopping
                            self.stop() # This will set self.is_recording = False
                            if not self.is_recording : # Successfully stopped
                                if self.speech_end_silence_start != 0:
                                    self.speech_end_silence_start = 0
                                    if self.on_turn_detection_stop: self._run_callback(self.on_turn_detection_stop)
                            else: failed_stop_attempt = True # Stop was too soon
                            self.awaiting_speech_end = False
                
                if not self.is_recording and was_recording: # Just stopped
                    self.stop_recording_on_voice_deactivity = False # Reset flag
                
                if time.time() - self.silero_check_time > 0.1: self.silero_check_time = 0 # Reset Silero check timer

                # Wake Word Timeout (if wake word was detected but no speech followed)
                if wakeword_detected_time and not self.is_recording and (time.time() - wakeword_detected_time > self.wake_word_timeout):
                    logger.info("Wake word timeout: No speech detected after wake word.")
                    wakeword_detected_time = 0 # Reset
                    if self.wakeword_detected and self.on_wakeword_timeout: self._run_callback(self.on_wakeword_timeout)
                    self.wakeword_detected = False
                    self.listen_start = 0 # Reset listen_start to go back to initial wake word mode or inactive
                
                was_recording = self.is_recording # Update for next iteration

                if self.is_recording and not failed_stop_attempt:
                    self.frames.append(data_bytes) # Add current (speech or silence) chunk to active recording
                
                if not self.is_recording or self.speech_end_silence_start: # If not recording or in post-speech silence phase
                    self.audio_buffer.append(data_bytes) # Keep buffering for pre-roll

        except Exception as e:
            if not self.interrupt_stop_event.is_set(): # Log only if not intentional interruption
                logger.error(f"Unhandled exception in _recording_worker: {e}", exc_info=True)
            # No raise here to allow cleanup, thread will exit.
        finally:
             if self.use_extended_logging: logger.debug('Debug: Exiting _recording_worker method')


    def _realtime_worker(self):
        try:
            logger.debug('Starting MLX realtime worker')
            if not self.enable_realtime_transcription:
                return

            last_transcription_time = time.time()
            min_audio_len_for_realtime_samples = int(0.2 * self.sample_rate) # e.g., 200ms of audio

            while self.is_running:
                if self.is_recording:
                    while (time.time() - last_transcription_time) < self.realtime_processing_pause:
                        time.sleep(0.001)
                        if not self.is_running or not self.is_recording: break
                    if not self.is_running or not self.is_recording: continue
                    
                    if self.awaiting_speech_end: # If VAD thinks speech ended, pause realtime
                        time.sleep(0.01)
                        continue 
                    
                    last_transcription_time = time.time()

                    # Construct audio_for_rt from self.frames (which are bytes)
                    audio_for_rt_np = None
                    if self.frames:
                        # Create a copy of frames to avoid issues if self.frames is modified by another thread (unlikely here but good practice)
                        current_frames_for_rt = list(self.frames) 
                        if current_frames_for_rt:
                            audio_bytes_concat = b''.join(current_frames_for_rt)
                            if len(audio_bytes_concat) >= (min_audio_len_for_realtime_samples * 2): # Check if enough bytes for min duration
                                audio_for_rt_np = np.frombuffer(audio_bytes_concat, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
                            # else: logger.debug(f"Not enough audio for realtime: {len(audio_bytes_concat)} bytes")
                    
                    if audio_for_rt_np is not None and len(audio_for_rt_np) > 0:
                        # Normalize if self.normalize_audio is set (MLXTranscriber also does normalization)
                        # if self.normalize_audio: audio_for_rt_np = self._normalize_audio(audio_for_rt_np)

                        self.mlx_transcriber.add_audio_chunk(audio_for_rt_np, is_last=False)
                        result = self.mlx_transcriber.get_result(timeout=0.05) # Shorter timeout for quick check
                        
                        if result and (time.time() - self.recording_start_time > self.init_realtime_after_seconds):
                            text = result.get('text', '') # Full text so far from this stream segment
                            new_text_segment = result.get('new_text', '') # Text from the last processed internal chunk

                            # Update self.realtime_transcription_text with the full current text from MLX
                            self.realtime_transcription_text = text 
                            
                            # Use self.realtime_transcription_text for stabilization logic
                            self.text_storage.append(self.realtime_transcription_text)
                            if len(self.text_storage) > 20: self.text_storage.pop(0) # Limit history

                            if len(self.text_storage) >= 2:
                                last_two_texts = self.text_storage[-2:]
                                common_prefix = os.path.commonprefix([last_two_texts[0], last_two_texts[1]])
                                if len(common_prefix) >= len(self.realtime_stabilized_safetext):
                                    self.realtime_stabilized_safetext = common_prefix
                            
                            final_display_text = ""
                            match_pos = self._find_tail_match_in_text(self.realtime_stabilized_safetext, self.realtime_transcription_text)
                            if match_pos < 0 : # No good match, or texts diverged too much
                                final_display_text = self.realtime_stabilized_safetext if self.realtime_stabilized_safetext else self.realtime_transcription_text
                            else:
                                final_display_text = self.realtime_stabilized_safetext + self.realtime_transcription_text[match_pos:]

                            if self.on_realtime_transcription_stabilized:
                                self._run_callback(self.on_realtime_transcription_stabilized, self._preprocess_output(final_display_text, True))
                            if self.on_realtime_transcription_update: # Send the latest full text
                                self._run_callback(self.on_realtime_transcription_update, self._preprocess_output(self.realtime_transcription_text, True))
                    # else: time.sleep(0.01) # No frames or not enough audio for RT processing
                else: # if not self.is_recording
                    self.realtime_stabilized_text = "" # Clear stabilized text when not recording
                    self.realtime_stabilized_safetext = ""
                    self.text_storage.clear()
                    time.sleep(TIME_SLEEP) 
        except Exception as e:
            logger.error(f"Unhandled exception in _realtime_worker: {e}", exc_info=True)
            # No raise here, allow thread to exit gracefully if is_running becomes false.

    def _is_silero_speech(self, data_bytes): # Expects bytes
        # Silero VAD expects 16kHz float32 audio
        # Data from audio_queue is already at self.sample_rate (e.g. 16kHz) and bytes
        if self.silero_working : return self.is_silero_speech_active # Avoid concurrent calls
        self.silero_working = True
        try:
            audio_chunk_np = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
            # Silero model expects a PyTorch tensor
            audio_tensor = torch.from_numpy(audio_chunk_np)
            vad_prob = self.silero_vad_model(audio_tensor, self.sample_rate).item()
            currently_is_speech = vad_prob > (1 - self.silero_sensitivity)

            if currently_is_speech and not self.is_silero_speech_active and self.use_extended_logging:
                logger.info(f"{bcolors.OKGREEN}Silero VAD detected speech{bcolors.ENDC}")
            elif not currently_is_speech and self.is_silero_speech_active and self.use_extended_logging:
                 logger.info(f"{bcolors.WARNING}Silero VAD detected silence{bcolors.ENDC}")
            self.is_silero_speech_active = currently_is_speech
        except Exception as e_silero:
            logger.error(f"Error in Silero VAD processing: {e_silero}", exc_info=True)
            # Default to previous state or conservative (no speech) on error
            # self.is_silero_speech_active = False 
        finally:
            self.silero_working = False
        return self.is_silero_speech_active

    def _is_webrtc_speech(self, data_bytes, all_frames_must_be_true=False): # Expects bytes
        # WebRTC VAD expects 16kHz, 16-bit PCM, 10/20/30 ms frames
        # data_bytes should be a chunk from audio_queue, e.g., 30ms (960 bytes at 16kHz)
        speech_str = f"{bcolors.OKGREEN}WebRTC VAD detected speech{bcolors.ENDC}"
        silence_str = f"{bcolors.WARNING}WebRTC VAD detected silence{bcolors.ENDC}"

        frame_duration_ms = 10 # WebRTC supports 10, 20, 30 ms
        samples_per_frame = int(self.sample_rate * (frame_duration_ms / 1000.0))
        bytes_per_frame = samples_per_frame * 2 # 16-bit = 2 bytes

        if len(data_bytes) % bytes_per_frame != 0:
            # This can happen if VAD chunk size from audio_queue isn't a multiple of WebRTC frame size
            # Pad or truncate, or adjust VAD chunk size to be a multiple.
            # For now, log a warning and process what we can.
            # logger.warning(f"WebRTC: data_bytes length {len(data_bytes)} not multiple of bytes_per_frame {bytes_per_frame}")
            pass # Or process only full frames

        num_frames = len(data_bytes) // bytes_per_frame
        if num_frames == 0: return self.is_webrtc_speech_active # Not enough data for a frame

        speech_frames_count = 0
        for i in range(num_frames):
            start_byte = i * bytes_per_frame
            end_byte = start_byte + bytes_per_frame
            frame = data_bytes[start_byte:end_byte]
            if len(frame) == bytes_per_frame: # Ensure frame is correct size
                if self.webrtc_vad_model.is_speech(frame, self.sample_rate):
                    speech_frames_count += 1
                    if not all_frames_must_be_true: # Any speech frame triggers detection
                        if not self.is_webrtc_speech_active and self.use_extended_logging: logger.info(speech_str)
                        self.is_webrtc_speech_active = True
                        return True
            # else: logger.warning(f"WebRTC: Short frame encountered. Length: {len(frame)}")

        currently_is_speech = (speech_frames_count == num_frames) if all_frames_must_be_true else (speech_frames_count > 0)
        
        if currently_is_speech and not self.is_webrtc_speech_active and self.use_extended_logging: logger.info(speech_str)
        elif not currently_is_speech and self.is_webrtc_speech_active and self.use_extended_logging: logger.info(silence_str)
        
        self.is_webrtc_speech_active = currently_is_speech
        return self.is_webrtc_speech_active

    def _check_voice_activity(self, data_bytes): # Expects bytes
        # Update WebRTC state first
        self._is_webrtc_speech(data_bytes) 
        # Then, if WebRTC detected speech (or if we want Silero to run always for more robustness)
        if self.is_webrtc_speech_active: # Or simply run Silero check regardless
            if not self.silero_working: # Avoid concurrent if already running
                # Run Silero check in a new thread to not block this VAD logic path
                threading.Thread(target=self._is_silero_speech, args=(data_bytes,), daemon=True).start()

    def clear_audio_queue(self): # Keep as is
        self.audio_buffer.clear()
        try:
            while True: self.audio_queue.get_nowait()
        except queue.Empty: pass # mp.Queue specific
        except Exception: pass # General catch for other queue types if any

    def _is_voice_active(self): # Keep as is
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state): # Keep as is
        if new_state == self.state: return
        old_state = self.state
        self.state = new_state
        logger.info(f"State changed from '{old_state}' to '{new_state}'")

        if old_state == "listening" and self.on_vad_detect_stop: self._run_callback(self.on_vad_detect_stop)
        elif old_state == "wakeword" and self.on_wakeword_detection_end: self._run_callback(self.on_wakeword_detection_end)

        if new_state == "listening":
            if self.on_vad_detect_start: self._run_callback(self.on_vad_detect_start)
            self._set_spinner("speak now"); halo_interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start: self._run_callback(self.on_wakeword_detection_start)
            self._set_spinner(f"say {self.wake_words if self.wake_words else 'wakeword'}"); halo_interval = 500 # Show generic if no specific words
        elif new_state == "transcribing":
            self._set_spinner("transcribing"); halo_interval = 50
        elif new_state == "recording":
            self._set_spinner("recording"); halo_interval = 100
        elif new_state == "inactive" and self.spinner and self.halo:
            self.halo.stop(); self.halo = None; return # Return early
        
        if self.spinner and self.halo : self.halo._interval = halo_interval


    def _set_spinner(self, text): # Keep as is
        if self.spinner:
            if self.halo is None: self.halo = halo.Halo(text=text); self.halo.start()
            else: self.halo.text = text

    def _preprocess_output(self, text_input, preview=False): # Keep as is, text_input can be string or list of segments
        # MLXTranscriber returns a single string, so this simplifies.
        # If it were segments, the original logic for joining segments would be needed.
        # text = " ".join(seg.text for seg in text_input).strip() # If text_input were segments
        text = str(text_input).strip() # Assuming text_input is already a string
        text = re.sub(r'\s+', ' ', text)

        if self.ensure_sentence_starting_uppercase and text:
            text = text[0].upper() + text[1:]
        
        if not preview and self.ensure_sentence_ends_with_period and text and text[-1].isalnum():
            text += '.'
        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10): # Keep as is
        if len(text1) < length_of_match or len(text2) < length_of_match: return -1
        target_substring = text1[-length_of_match:]
        for i in range(len(text2) - length_of_match + 1):
            current_substring = text2[len(text2) - i - length_of_match : len(text2) - i]
            if current_substring == target_substring: return len(text2) - i
        return -1

    def _on_realtime_transcription_stabilized(self, text): # Internal helper, keep as is
        if self.on_realtime_transcription_stabilized and self.is_recording:
            self._run_callback(self.on_realtime_transcription_stabilized, text)

    def _on_realtime_transcription_update(self, text): # Internal helper, keep as is
        if self.on_realtime_transcription_update and self.is_recording:
            self._run_callback(self.on_realtime_transcription_update, text)

    def __enter__(self): # Keep as is
        return self

    def __exit__(self, exc_type, exc_value, traceback): # Keep as is
        self.shutdown()
