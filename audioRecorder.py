import sounddevice as sd
import soundfile as sf
import threading
from queue import Queue
import serial
import sys

class AudioRecorder:
    def __init__(self, mic_id, sample_rate, channels, arduino_port):
        # Microphone ID
        self.mic_id = mic_id
        # Sample rate
        self.sample_rate = sample_rate
        # Number of channels
        self.channels = channels
        # Recording flag
        self.recording = False
        # Stop request flag
        self.stop_requested = False
        # Start the audio queue
        self.mic_queue = Queue()  
        
        # Connection to the Arduino board via serial communication
        self._serial = serial.Serial(arduino_port, 115200)
        

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.recording:
            self.mic_queue.put(indata.copy())

    def start_recording(self, filepath):
        # Output audio file
        self.filepath = filepath
        if not self.recording:
            self.recording = True
            self.stop_requested = False  # Reset the stop_requested flag
            self._send_pulse_to_arduino()
            self.mic_queue = Queue()  # Reset the audio queue
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            # Set a timer to stop recording after ... seconds
            self.stop_timer = threading.Timer(6.0, self.stop_recording)
            self.stop_timer.start()

    def stop_recording(self):
        if self.recording:
            self.stop_requested = True
            self._send_pulse_to_arduino()
            self.recording_thread.join()  # Wait for the recording thread to finish
            self.recording = False
            if hasattr(self, 'stop_timer'):
                self.stop_timer.cancel()  # Cancel the timer if it's still running

    def _send_pulse_to_arduino(self):
        # Message to send to Arduino
        message = "P"
        # Send the message to Arduino
        self._serial.write(message.encode())

    def _record_audio(self):
        with sf.SoundFile(self.filepath, mode='x', samplerate=self.sample_rate, channels=self.channels, subtype=None) as file:
            with sd.InputStream(samplerate=self.sample_rate, device=self.mic_id, channels=self.channels, callback=self.callback):
                try:
                    while not self.stop_requested:
                        file.write(self.mic_queue.get())
                except RuntimeError as re:
                    print(f"{re}. If recording was stopped by the user, then this can be ignored")
                finally:
                    self.recording = False