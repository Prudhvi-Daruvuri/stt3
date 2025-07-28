"""
Faster Whisper Transcription Worker Module

This module contains the TranscriptionWorker class that handles
speech-to-text transcription using the faster_whisper library.
Extracted from audio_recorder.py for better code organization.

Author: Kolja Beigel
"""

from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch.multiprocessing as mp
import signal as system_signal
import faster_whisper
import numpy as np
import traceback
import threading
import logging
import struct
import base64
import queue
import time
import os
import soundfile as sf

# Named logger for this module.
logger = logging.getLogger("realtimestt")

TIME_SLEEP = 0.02
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter, normalize_audio):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path
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
                if self.conn.poll(0.01):  # Increased from 0.01 to 0.5 seconds
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

        logging.info(f"Initializing faster_whisper main transcription model {self.model_path}")

        try:
            model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index,
                download_root=self.download_root,
            )
            # Create a short dummy audio array, for example 1 second of silence at 16 kHz
            if self.batch_size > 0:
                model = BatchedInferencePipeline(model=model)

            # Run a warm-up transcription
            current_dir = os.path.dirname(os.path.realpath(__file__))
            warmup_audio_path = os.path.join(
                current_dir, "warmup_audio.wav"
            )
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            segments, info = model.transcribe(warmup_audio_data, language="en", beam_size=1)
            model_warmup_transcription = " ".join(segment.text for segment in segments)
        except Exception as e:
            logging.exception(f"Error initializing main faster_whisper transcription model: {e}")
            raise

        self.ready_event.set()
        logging.debug("Faster_whisper main speech to text transcription model initialized successfully")

        # Start the polling thread
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language, use_prompt = self.queue.get(timeout=0.1)
                    try:
                        logging.debug(f"Transcribing audio with language {language}")
                        start_t = time.time()

                        # normalize audio to -0.95 dBFS
                        if audio is not None and audio .size > 0:
                            if self.normalize_audio:
                                peak = np.max(np.abs(audio))
                                if peak > 0:
                                    audio = (audio / peak) * 0.95
                        else:
                            logging.error("Received None audio for transcription")
                            self.conn.send(('error', "Received None audio for transcription"))
                            continue

                        prompt = None
                        if use_prompt:
                            prompt = self.initial_prompt if self.initial_prompt else None

                        if self.batch_size > 0:
                            segments, info = model.transcribe(
                                audio,
                                language=language if language else None,
                                beam_size=self.beam_size,
                                initial_prompt=prompt,
                                suppress_tokens=self.suppress_tokens,
                                batch_size=self.batch_size, 
                                vad_filter=self.faster_whisper_vad_filter
                            )
                        else:
                            segments, info = model.transcribe(
                                audio,
                                language=language if language else None,
                                beam_size=self.beam_size,
                                initial_prompt=prompt,
                                suppress_tokens=self.suppress_tokens,
                                vad_filter=self.faster_whisper_vad_filter
                            )
                        elapsed = time.time() - start_t
                        transcription = " ".join(seg.text for seg in segments).strip()
                        logging.debug(f"Final text detected with main model: {transcription} in {elapsed:.4f}s")
                        self.conn.send(('success', (transcription, info)))
                    except Exception as e:
                        logging.error(f"General error in transcription: {e}", exc_info=True)
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}", exc_info=True)
        finally:
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish
