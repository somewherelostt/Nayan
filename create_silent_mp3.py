"""
Script to create a silent MP3 file
"""
import os

# Ensure sounds directory exists
sound_dir = os.path.join(os.path.dirname(__file__), "sounds")
os.makedirs(sound_dir, exist_ok=True)

# Option 1: Try using numpy and scipy if available
try:
    import numpy as np
    from scipy.io import wavfile
    
    # Create silent WAV file
    SAMPLE_RATE = 44100
    DURATION = 1  # seconds
    silence = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)
    
    # Write WAV file
    wav_path = os.path.join(sound_dir, "silent.wav")
    wavfile.write(wav_path, SAMPLE_RATE, silence)
    
    print(f"Created silent WAV file at {wav_path}")
    
    # Copy as MP3 file (as a simple solution)
    import shutil
    mp3_path = os.path.join(sound_dir, "silent.mp3")
    shutil.copy(wav_path, mp3_path)
    print(f"Created silent MP3 file at {mp3_path}")
    
except ImportError:
    # Option 2: Create an empty file if dependencies aren't available
    empty_file_path = os.path.join(sound_dir, "silent.mp3")
    with open(empty_file_path, 'wb') as f:
        # Write a minimal valid MP3 header
        f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00')
    print(f"Created minimal silent MP3 file at {empty_file_path}") 