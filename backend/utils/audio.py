"""
Audio management utilities for Nayan
"""
import os
import queue
import threading
import time
import logging
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: sounddevice or soundfile modules not available. Sound will be disabled.")
    AUDIO_AVAILABLE = False
import requests


class AudioManager:
    def __init__(self):
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AudioManager")
        
        # Audio queue for playing sounds
        self.audio_queue = queue.Queue()
        
        # Sound directory
        self.sound_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sounds")
        os.makedirs(self.sound_dir, exist_ok=True)
        
        # Test if audio playback works
        self.audio_working = AUDIO_AVAILABLE
        
        # Track last played sound
        self.last_played_sound = None
        
        # Start audio thread if available
        if self.audio_working:
            self.audio_thread = threading.Thread(target=self.audio_worker)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            self.logger.info("Audio thread started successfully")
        else:
            self.logger.warning("Audio playback is disabled")
            
        # Ensure sound files exist
        self.initialize_sounds()
        
        # Try to test play a sound
        if self.audio_working:
            self.test_audio_playback()
    
    def test_audio_playback(self):
        """Test if audio playback works"""
        silent_file = os.path.join(self.sound_dir, "silent.mp3")
        if os.path.exists(silent_file):
            try:
                data, fs = sf.read(silent_file)
                sd.play(data, fs)
                sd.wait()
                self.logger.info("Audio playback test successful")
            except Exception as e:
                self.logger.error(f"Audio playback test failed: {e}")
                self.audio_working = False
        else:
            self.logger.warning("No test audio file available")
    
    def initialize_sounds(self):
        """Download necessary sound files"""
        self.logger.info("Initializing sound files")
        
        # Use GitHub as a more reliable alternative to soundbible.com
        self.download_sound_if_needed("proximity_alert", "https://github.com/anars/blank-audio/raw/master/250-milliseconds-of-silence.mp3")
        self.download_sound_if_needed("object_detected", "https://github.com/anars/blank-audio/raw/master/250-milliseconds-of-silence.mp3")
        
        # Also try to download actual sounds from alternative sources
        try:
            # Try to download a real alert sound from a reliable source
            alert_url = "https://cdn.pixabay.com/download/audio/2022/03/15/audio_c8c8a73467.mp3"
            self.download_sound_if_needed("proximity_alert", alert_url)
            
            # Try to download an object detection sound
            detect_url = "https://cdn.pixabay.com/download/audio/2021/08/04/audio_c518b3f49d.mp3"
            self.download_sound_if_needed("object_detected", detect_url)
            
            self.logger.info("Downloaded actual sound files")
        except Exception as e:
            self.logger.error(f"Failed to download actual sounds: {e}")
        
        # Create fallback silent sounds if downloads fail
        self.create_silent_fallback("proximity_alert")
        self.create_silent_fallback("object_detected")
        
        # List available sounds
        self.list_available_sounds()
    
    def list_available_sounds(self):
        """List all available sound files"""
        try:
            sound_files = [f for f in os.listdir(self.sound_dir) if f.endswith('.mp3')]
            self.logger.info(f"Available sound files: {sound_files}")
            for sound in sound_files:
                path = os.path.join(self.sound_dir, sound)
                size = os.path.getsize(path)
                self.logger.info(f"Sound file: {sound}, Size: {size} bytes")
        except Exception as e:
            self.logger.error(f"Error listing sound files: {e}")
    
    def download_sound_if_needed(self, name, url):
        """Download sound files if they don't exist"""
        file_path = os.path.join(self.sound_dir, f"{name}.mp3")
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:  # If file doesn't exist or is too small
            try:
                self.logger.info(f"Downloading sound {name} from {url}")
                # Set verify=False only if absolutely necessary and as a temporary solution
                response = requests.get(url, verify=True, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                self.logger.info(f"Downloaded {name} sound file ({len(response.content)} bytes)")
                print(f"Downloaded {name} sound file")
            except Exception as e:
                self.logger.error(f"Failed to download sound {name}: {e}")
                print(f"Failed to download sound: {e}")
    
    def create_silent_fallback(self, name):
        """Create a silent audio file as fallback if download fails"""
        file_path = os.path.join(self.sound_dir, f"{name}.mp3")
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:  # If file doesn't exist or is too small
            try:
                self.logger.info(f"Creating fallback for {name}")
                # Try to use existing silent file first
                silent_file = os.path.join(self.sound_dir, "silent.mp3")
                if os.path.exists(silent_file) and os.path.getsize(silent_file) > 100:
                    # Copy silent file
                    import shutil
                    shutil.copy(silent_file, file_path)
                    self.logger.info(f"Created silent fallback for {name}")
                    print(f"Created silent fallback for {name}")
                else:
                    # Create empty MP3 file with minimal header
                    with open(file_path, 'wb') as f:
                        # Write a minimal valid MP3 header
                        f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00')
                    self.logger.info(f"Created minimal silent MP3 file for {name}")
                    print(f"Created minimal silent MP3 file for {name}")
            except Exception as e:
                self.logger.error(f"Failed to create silent fallback: {e}")
                print(f"Failed to create silent fallback: {e}")
    
    def audio_worker(self):
        """Background thread to handle audio alerts"""
        while True:
            try:
                if not self.audio_queue.empty() and self.audio_working:
                    sound_file = self.audio_queue.get()
                    self.last_played_sound = os.path.basename(sound_file)
                    self.logger.info(f"Playing sound: {sound_file}")
                    
                    if os.path.exists(sound_file):
                        try:
                            data, fs = sf.read(sound_file)
                            sd.play(data, fs)
                            sd.wait()
                            self.logger.info(f"Finished playing sound: {sound_file}")
                        except Exception as e:
                            self.logger.error(f"Error playing sound {sound_file}: {e}")
                    else:
                        self.logger.warning(f"Sound file does not exist: {sound_file}")
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Audio worker error: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def play_sound(self, sound_name):
        """Add sound to the audio queue"""
        sound_file = os.path.join(self.sound_dir, f"{sound_name}.mp3")
        
        # Log attempt to play sound
        self.logger.info(f"Attempting to play sound: {sound_name}")
        print(f"SOUND: Playing {sound_name}")
        
        if os.path.exists(sound_file):
            file_size = os.path.getsize(sound_file)
            self.logger.info(f"Sound file exists: {sound_file} ({file_size} bytes)")
            
            if file_size > 100:  # Check if file is not empty
                if self.audio_working:
                    self.audio_queue.put(sound_file)
                    return True
                else:
                    self.logger.warning("Audio playback is disabled")
                    return False
            else:
                self.logger.warning(f"Sound file is too small: {sound_file} ({file_size} bytes)")
                return False
        else:
            self.logger.warning(f"Sound file not found: {sound_file}")
            print(f"Sound file {sound_file} not found")
            return False
    
    def get_last_played_sound(self):
        """Return the last played sound for debugging"""
        return self.last_played_sound 