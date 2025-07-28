"""
Parakeet Transcription Worker Module

This module contains the ParakeetTranscriptionWorker class that handles
speech-to-text transcription using the NVIDIA NeMo Parakeet ASR model.
Created to provide an alternative backend to faster_whisper.

Author: Based on Modal Parakeet example and RealtimeSTT architecture
"""

import torch
import torch.multiprocessing as mp
import signal as system_signal
import numpy as np
import traceback
import threading
import logging
import struct
import base64
import queue
import time
import os

# Named logger for this module.
logger = logging.getLogger("realtimestt")

TIME_SLEEP = 0.02
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class ParakeetTranscriptionWorker:
    def __init__(self, conn, stdout_pipe, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter, normalize_audio):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path or "nvidia/parakeet-tdt-0.6b-v2"  # Default Parakeet model
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.batch_size = batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        while not self.shutdown_event.is_set():
            try:
                # Use a longer timeout to reduce polling frequency
                if self.conn.poll(0.01):
                    data = self.conn.recv()
                    self.queue.put(data)
                else:
                    # Sleep only if no data, but use a shorter sleep
                    time.sleep(TIME_SLEEP)
            except Exception as e:
                logging.error(f"Error receiving data from connection: {e}", exc_info=True)
                time.sleep(TIME_SLEEP)

    def run(self):
        if __name__ == "__main__":
             system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
             __builtins__['print'] = self.custom_print

        logging.info(f"Initializing Parakeet ASR model {self.model_path}")

        try:
            # Import NeMo here to avoid dependency issues if not installed
            try:
                import nemo.collections.asr as nemo_asr
                # Silence chatty logs from nemo
                logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)
                logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
            except ImportError as e:
                error_msg = (
                    "NeMo toolkit not found. To use Parakeet backend, install with: "
                    "pip install nemo_toolkit[asr]"
                )
                logging.error(error_msg)
                raise ImportError(error_msg) from e

            # Load the Parakeet model - use the specific Parakeet model name
            # Parakeet models have different names than Whisper models
            parakeet_model_name = "nvidia/parakeet-tdt-0.6b-v2"  # Default Parakeet model
            
            # If a specific Parakeet model is provided, use it; otherwise use default
            if self.model_path and "parakeet" in self.model_path.lower():
                parakeet_model_name = self.model_path
            elif self.model_path and self.model_path not in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]:
                # If it's not a standard Whisper model name, assume it's a custom Parakeet model
                parakeet_model_name = self.model_path
            
            logging.info(f"Loading Parakeet model: {parakeet_model_name}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=parakeet_model_name
            )
            
            # Move model to specified device if available
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                logging.info(f"Parakeet model moved to CUDA device {self.gpu_device_index}")
            else:
                logging.info("Parakeet model running on CPU")

            # Run a warm-up transcription with dummy audio
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
            with self._suppress_nemo_output():
                warmup_output = self.model.transcribe([dummy_audio])
                warmup_text = warmup_output[0].text if warmup_output else ""
            
            logging.debug(f"Parakeet model warmed up with dummy transcription: '{warmup_text}'")
            
        except Exception as e:
            logging.exception(f"Error initializing Parakeet ASR model: {e}")
            raise

        self.ready_event.set()
        logging.debug("Parakeet ASR model initialized successfully")

        # Start the polling thread
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language, use_prompt = self.queue.get(timeout=0.1)
                    try:
                        logging.debug(f"Transcribing audio with Parakeet (language: {language})")
                        start_t = time.time()

                        # Convert audio bytes to numpy array (int16 -> float32)
                        if audio is not None and len(audio) > 0:
                            if isinstance(audio, bytes):
                                audio_data = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
                            elif isinstance(audio, np.ndarray):
                                if audio.dtype == np.int16:
                                    audio_data = audio.astype(np.float32)
                                else:
                                    audio_data = audio
                            else:
                                logging.error(f"Unsupported audio data type: {type(audio)}")
                                self.conn.send(('error', f"Unsupported audio data type: {type(audio)}"))
                                continue

                            # Normalize audio if requested
                            if self.normalize_audio and len(audio_data) > 0:
                                peak = np.max(np.abs(audio_data))
                                if peak > 0:
                                    audio_data = (audio_data / peak) * 0.95
                        else:
                            logging.error("Received None or empty audio for transcription")
                            self.conn.send(('error', "Received None or empty audio for transcription"))
                            continue

                        # Perform transcription with Parakeet
                        with self._suppress_nemo_output():
                            output = self.model.transcribe([audio_data])
                            transcription = output[0].text if output and len(output) > 0 else ""

                        elapsed = time.time() - start_t
                        logging.debug(f"Parakeet transcription: '{transcription}' in {elapsed:.4f}s")
                        
                        # Create mock info object similar to faster_whisper format
                        mock_info = type('MockInfo', (), {
                            'language': language or 'en',
                            'language_probability': 0.9  # Parakeet doesn't provide this, use default
                        })()
                        
                        self.conn.send(('success', (transcription, mock_info)))
                        
                    except Exception as e:
                        logging.error(f"General error in Parakeet transcription: {e}", exc_info=True)
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Parakeet transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}", exc_info=True)
        finally:
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish

    def _suppress_nemo_output(self):
        """Context manager to suppress NeMo's verbose output during transcription"""
        import contextlib
        import sys
        import os
        
        @contextlib.contextmanager
        def suppress_stdout_stderr():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        
        return suppress_stdout_stderr()
