"""
Test script for audio and speech functionality
"""
import os
import sys
import time

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.audio import AudioManager
from backend.utils.speech import SpeechManager

def main():
    """Test audio and speech functionality"""
    print("Testing audio and speech functionality...")
    
    # Initialize managers
    print("\nInitializing speech manager...")
    speech_manager = SpeechManager()
    
    print("\nInitializing audio manager...")
    audio_manager = AudioManager()
    
    # Wait for initialization
    time.sleep(2)
    
    # Test speech
    print("\nTesting speech announcements...")
    speech_manager.announce("This is a test of the speech system", priority=True)
    time.sleep(3)
    
    # Test sounds
    print("\nTesting sound playback...")
    print("Playing proximity alert sound...")
    audio_manager.play_sound("proximity_alert")
    time.sleep(3)
    
    print("Playing object detection sound...")
    audio_manager.play_sound("object_detected")
    time.sleep(3)
    
    # List available sounds
    print("\nAvailable sound files:")
    sound_dir = audio_manager.sound_dir
    try:
        sound_files = [f for f in os.listdir(sound_dir) if f.endswith('.mp3')]
        for sound in sound_files:
            path = os.path.join(sound_dir, sound)
            size = os.path.getsize(path)
            print(f"Sound file: {sound}, Size: {size} bytes")
    except Exception as e:
        print(f"Error listing sound files: {e}")
    
    # Test final announcement
    speech_manager.announce("Audio and speech test complete", priority=True)
    time.sleep(3)
    
    print("\nTest complete. If you didn't hear any sounds or speech, check the logs above for errors.")

if __name__ == "__main__":
    main() 