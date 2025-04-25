"""
Speech management utilities for Nayan
"""
import pyttsx3
import threading
import time
import logging
import os


class SpeechManager:
    def __init__(self):
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SpeechManager")
        
        # Initialize the text-to-speech engine with proper error handling
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed of speech
            
            # Set up voice options
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)  # Prefer female voice
                self.logger.info(f"TTS initialized with voice: {self.engine.getProperty('voice')}")
            else:
                self.logger.warning("No voices found for TTS engine")
            
            # Test the TTS engine
            def test_tts():
                try:
                    self.engine.say("TTS test")
                    self.engine.runAndWait()
                    return True
                except Exception as e:
                    self.logger.error(f"TTS test failed: {e}")
                    return False
            
            self.tts_working = test_tts()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
            self.tts_working = False
        
        # Speech queue and status
        self.speech_queue = []
        self.speaking = False
        self.last_announcement = ""
        
        # Start the speech thread
        self.speech_thread = threading.Thread(target=self.speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
    
    def speech_worker(self):
        """Background thread to handle speech without blocking the main thread"""
        while True:
            try:
                if self.speech_queue and not self.speaking and self.tts_working:
                    self.speaking = True
                    announcement = self.speech_queue.pop(0)
                    self.last_announcement = announcement
                    self.logger.info(f"Speaking: {announcement}")
                    
                    if self.engine:
                        self.engine.say(announcement)
                        self.engine.runAndWait()
                    
                    self.speaking = False
                    
                    # Print announcements to console for debugging
                    print(f"SPEECH: {announcement}")
                elif self.speech_queue and not self.tts_working:
                    # If TTS is not working, just log the announcement
                    announcement = self.speech_queue.pop(0)
                    self.last_announcement = announcement
                    self.logger.info(f"TTS disabled, would have said: {announcement}")
                    print(f"SPEECH (TTS disabled): {announcement}")
                
                # Sleep briefly to prevent CPU hogging
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in speech worker: {e}")
                self.speaking = False
                time.sleep(0.5)
    
    def announce(self, text, priority=False):
        """Add text to speech queue, with optional priority flag"""
        if not text:
            return
            
        self.logger.info(f"Queueing announcement: {text} (priority: {priority})")
        
        if priority:
            self.speech_queue.insert(0, text)
        else:
            self.speech_queue.append(text)
        
        # Return the text for API responses
        return text
        
    def get_last_announcement(self):
        """Return the last announcement for debugging"""
        return self.last_announcement 