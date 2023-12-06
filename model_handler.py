import vosk
from utils_vad import OnnxWrapper
from pathlib import Path
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer
from misc import Command

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound



class SpeechRecognizer:
    """Handles vosk and vad speech to text"""
    def __init__(self, model_path, sample_rate, chunk_size):
        """Class Constructor"""
        self.vad = OnnxWrapper(str(Path(model_path, 'silero_vad.onnx')))
        self.audio_chunks = []
        self.speech_start_idx = 0
        self.speech_end_idx = 0
        self.start_speech = False
        self.rate = sample_rate
        offset_duration = 0.5 # in seconds
        self.chunk_offset = 2*int(np.ceil((self.rate*offset_duration)/chunk_size))

        self.numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",]
        
        self.unknown_word = "[unk]"

        model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(model, self.rate, json.dumps([
        # System commands
        "start", "stop", "robot", "execution", "move", "go", "set mode", "continuous", "model", "step", "size", "tool", "open", "close", "rotate",
        "save", "home", "position", "load", "place", "the",
        # Directions
        "up", "down", "left", "right", "forward", "backward", "front", "back",
        # numbers
        *self.numbers,
        "minus", "negative",
        # cliport
        # unknown
        self.unknown_word]))
        
    
    def speech_to_text(self, data):
        """Convert speech to text using speech model recognizer"""
        words = []
        number = None
        # Detect Speech
        self.audio_chunks.append(data)
        audio_int16 = np.frombuffer(data, np.int16)
        audio_float32 = int2float(audio_int16)
        output = self.vad(audio_float32, self.rate)
        if output > 0.5:
            if not self.start_speech:
                self.speech_start_idx = len(self.audio_chunks) - 1
            self.start_speech = True
            self.speech_end_idx = len(self.audio_chunks) + self.chunk_offset
        else:
            if self.start_speech and len(self.audio_chunks) > self.speech_end_idx:
                start_idx = max(0, self.speech_start_idx - self.chunk_offset)
                self.start_speech = False
                speech = b''.join(self.audio_chunks[start_idx:])
                self.audio_chunks = []
                # Convert speech to text
                self.rec.AcceptWaveform(speech)
                words = json.loads(self.rec.FinalResult())["text"].split(' ')
                # Find number in word sequence (ONLY works for single numeric sequence)
                num_str = ""
                is_positive = True
                found_unknown = False
                for word in words:
                    if word in self.numbers:
                        num_str = num_str + f" {word}"
                    if word in ['minus', 'negative']:
                        is_positive = False
                    if self.unknown_word == word:
                        found_unknown = True
                        break
                if num_str != "":
                    try:
                        number = (2*is_positive - 1)*w2n.word_to_num(num_str)
                    except Exception:
                        number = None
                # Filter out instructions that contain unknown word
                if found_unknown:
                    words = []
                    number = None
                self.vad.reset_states()
        return words, number

class Recognizer:
    """Handles vosk and vad speech to text"""
    def __init__(self, model_path, sample_rate, debug_enabled = False):
        """Class Constructor"""
        self.vad = OnnxWrapper(str(Path(model_path, 'silero_vad.onnx')))
        self.audio_buffer = []
        self.start_speech = False
        self.time_since_last_speech = 0.0
        self.speech_end_interval = 0.5
        self.chunk_offset = 8
        self.rate = sample_rate
        self.debug_enabled = debug_enabled
        """Class Constructor"""
        self.vad = OnnxWrapper(str(Path(model_path, 'silero_vad.onnx')))
        self.audio_chunks = []
        self.speech_start_idx = 0
        self.speech_end_idx = 0
        self.start_speech = False
        self.rate = sample_rate
        offset_duration = 0.5 # in seconds
        
        model = vosk.Model(model_path)
        self.numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",]
        
        self.unknown_word = "[unk]"

        self.rec = vosk.KaldiRecognizer(model, self.rate, json.dumps([
        # System commands
        "start", "stop","panda", "robot", "execution", "move", "go", "set mode", "continuous", "model", "step", "size", "tool", "open", "close", "rotate",
        "save", "home", "position", "load", "place", "the",
        # Directions
        "up", "down", "left", "right", "forward", "backward", "front", "back",
        # numbers
        #*self.numbers,
        #"minus", "negative",
        # cliport
        # unknown
        self.unknown_word,
        # System commands
        "start", "stop", "panda", "robot", "recover", "move", "go", "mode", "distance", "direction", "step", "low", "medium", "high", "size", "tool", "open", "close", "grasp", "rotate",
        "list", "show", "task", "play", "do", "remove","again", "delete", "save", "home", "finish", "record", "gripper", "position", "spot", "other", "opposite", "counter", "and", "then","corner", "speed", "velocity",
        "take", "give", "new", "name", "return","drop","all","man",
        # Directions
        "up", "down", "left", "right", "forward", "backward", "front", "back",
        # colors
        "red", "blue", "green", "yellow",
        # unknown
        "[unk]"]))
        
    
    def speech_to_text1(self, data):
        """Convert speech to text using speech model recognizer"""
        # Detect Speech
        if not self.start_speech:
            self.audio_buffer.append(data)
        audio_int16 = np.frombuffer(data, np.int16)
        audio_float32 = int2float(audio_int16)
        output = self.vad(audio_float32, self.rate)

        if output > 0.5:
            self.debug("Speech Detected")
            self.time_since_last_speech = time.time()
            if not self.start_speech:
                self.start_speech = True
                # Add a few audio chunks before detecting speech
                speech = b''.join(self.audio_buffer[-self.chunk_offset:-1])
                self.rec.AcceptWaveform(speech)
                self.audio_buffer = []

        if self.start_speech:
            self.rec.AcceptWaveform(data)
            partial = json.loads(self.rec.PartialResult())['partial']
            if partial != '':
                self.debug(partial)

            if 'stop' in partial:
                self.rec.Reset()
                words = ['stop','panda']
                return words

        if self.start_speech and time.time() - self.time_since_last_speech > self.speech_end_interval:
            self.start_speech = False
            text = json.loads(self.rec.FinalResult())["text"]
            self.debug('Final result: {text}')
            words = text.split(' ')
            self.debug("Speech Ended")
            return words
        return None
    

    def speech_to_text(self, data):
        """Convert speech to text using speech model recognizer"""
        words = []
        number = None
        # Detect Speech
        self.audio_chunks.append(data)
        audio_int16 = np.frombuffer(data, np.int16)
        audio_float32 = int2float(audio_int16)
        output = self.vad(audio_float32, self.rate)
        if output > 0.5:
            if not self.start_speech:
                self.speech_start_idx = len(self.audio_chunks) - 1
            self.start_speech = True
            self.speech_end_idx = len(self.audio_chunks) + self.chunk_offset
        else:
            if self.start_speech and len(self.audio_chunks) > self.speech_end_idx:
                start_idx = max(0, self.speech_start_idx - self.chunk_offset)
                self.start_speech = False
                speech = b''.join(self.audio_chunks[start_idx:])
                self.audio_chunks = []
                # Convert speech to text
                self.rec.AcceptWaveform(speech)
                words = json.loads(self.rec.FinalResult())["text"].split(' ')
                # Find number in word sequence (ONLY works for single numeric sequence)
                num_str = ""
                is_positive = True
                found_unknown = False
                for word in words:
                    if word in self.numbers:
                        num_str = num_str + f" {word}"
                    if word in ['minus', 'negative']:
                        is_positive = False
                    if self.unknown_word == word:
                        found_unknown = True
                        break
                if num_str != "":
                    try:
                        number = (2*is_positive - 1)*w2n.word_to_num(num_str)
                    except Exception:
                        number = None
                # Filter out instructions that contain unknown word
                if found_unknown:
                    words = []
                    number = None
                self.vad.reset_states()
        return words, number

    
    






    def debug(self, text):
        if self.debug_enabled:
            print(text)

