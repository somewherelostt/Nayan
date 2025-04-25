"""
Core Nayan class for handling computer vision-based assistance
"""
import cv2
import numpy as np
import time
import pyttsx3
import threading
import queue
import os
import torch
from collections import deque
from ultralytics import YOLO
import sounddevice as sd
import soundfile as sf
import requests

from ..utils.audio import AudioManager
from ..utils.speech import SpeechManager
from ..models.model_loader import ModelLoader
from ..detectors.object_detector import ObjectDetector
from ..detectors.hazard_detector import HazardDetector
from ..detectors.environment_detector import EnvironmentDetector
from ..detectors.text_detector import TextDetector


class EnhancedNayan:
    def __init__(self, camera_index=0, use_sample_video=False, lazy_loading=True):
        # Initialize managers
        self.speech_manager = SpeechManager()
        self.audio_manager = AudioManager()
        
        # For lazy loading models to improve startup time
        self.lazy_loading = lazy_loading
        self.models_loaded = False
        
        # Initialize model loader but don't load models yet if lazy loading
        self.model_loader = ModelLoader()
        if not self.lazy_loading:
            self._load_models()
        else:
            # Initialize with None
            self.model = None
            self.midas = None
            self.transform = None
            self.device = self.model_loader.get_device()
            print("Using lazy loading for models to improve performance")
        
        # Initialize detectors
        self.object_detector = ObjectDetector()
        self.hazard_detector = HazardDetector()
        self.environment_detector = EnvironmentDetector()
        self.text_detector = TextDetector()
        
        # Initialize webcam or video file
        if use_sample_video:
            # Use a sample video file if available
            sample_path = os.path.join(os.path.dirname(__file__), '..', '..', 'sample_data', 'sample_video.mp4')
            if os.path.exists(sample_path):
                self.cap = cv2.VideoCapture(sample_path)
            else:
                # Create a blank video stream if sample doesn't exist
                self.cap = self._create_blank_video()
                print("Sample video not found, using blank video")
        else:
            # Try to open camera with specified index
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print(f"Cannot open camera with index {camera_index}, trying other indices...")
                # Try other camera indices
                for idx in range(5):
                    if idx == camera_index:
                        continue
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap.isOpened():
                        print(f"Successfully opened camera with index {idx}")
                        break
                
                # If still not opened, create a blank video
                if not self.cap.isOpened():
                    print("No camera found, using blank video")
                    self.cap = self._create_blank_video()
        
        # Improve camera buffering if available
        if isinstance(self.cap, cv2.VideoCapture):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
            
        # Get camera parameters
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Reference sizes for distance estimation (in cm)
        self.reference_objects = {
            "person": 170,  # Average height in cm
            "car": 150,     # Average height in cm
            "bottle": 20,   # Average height in cm
            "chair": 80,    # Average height in cm
            "dog": 50,      # Average height in cm
            "cat": 30       # Average height in cm
        }
        
        # Set detection parameters
        self.conf_threshold = 0.5  # Confidence threshold for detections
        self.last_speech_time = time.time()
        self.speech_cooldown = 3.0  # Time between speech announcements
        self.scene_description_cooldown = 15.0  # Time between scene descriptions
        self.last_scene_time = 0
        self.current_detections = []
        
        # Path planning parameters
        self.obstacle_memory = deque(maxlen=10)  # Remember recent obstacles
        self.safe_path = None
        
        # Depth map parameters
        self.depth_map = None
        self.depth_colormap = None
        self.depth_threshold = 2.0  # Meters - objects closer than this are considered immediate obstacles
        
        # Add a flag to skip depth processing sometimes for better performance
        self.skip_depth_count = 0
        self.skip_depth_frames = 3  # Process depth every N frames
        
        # OCR integration data
        self.ocr_active = False
        self.ocr_cooldown = 10.0  # Time between OCR scans
        self.last_ocr_time = 0
        
        # Navigation history
        self.location_history = deque(maxlen=100)  # Store recent locations for backtracking
        self.landmarks = {}  # Store named landmarks
        
        # Interactive mode parameters
        self.interactive_mode = False
        self.voice_commands = {
            "describe": self.cmd_describe_scene,
            "identify": self.cmd_identify_objects,
            "read": self.cmd_read_text,
            "navigate": self.cmd_navigate,
            "remember": self.cmd_remember_location,
            "locate": self.cmd_locate_landmark,
            "help": self.cmd_help
        }
        
        # Initialize sounds
        self.audio_manager.initialize_sounds()
        
        # Announce initialization
        self.speech_manager.announce("Enhanced Nayan system initialized")
    
    def _load_models(self):
        """Load ML models on demand"""
        try:
            print("Loading models...")
            start_time = time.time()
            # Load YOLO model
            self.model = self.model_loader.load_yolo_model()
            # Optionally load MiDaS model
            self.midas = self.model_loader.load_midas_model()
            self.transform = self.model_loader.get_midas_transform()
            self.models_loaded = True
            print(f"Models loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading models: {e}")
            # Set a flag to retry loading later
            self.models_loaded = False
    
    def _create_blank_video(self):
        """Create a blank video source when no camera is available"""
        # Define dimensions for the blank frame
        width = 640
        height = 480
        
        class BlankVideoCapture:
            def __init__(self, width, height):
                self.width = width
                self.height = height
                self.is_opened = True
                self.frame_count = 0
            
            def read(self):
                # Create a black frame with text
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                # Make it more interesting with a pulsing message
                self.frame_count += 1
                pulse = int(20 * np.sin(self.frame_count * 0.1) + 20)
                
                # Add text saying no camera found
                cv2.putText(frame, "No camera available", (self.width//4, self.height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add pulsing circle to indicate it's active
                cv2.circle(frame, (self.width//2, self.height//2 + 50), 
                          pulse, (0, 0, 255), -1)
                
                return True, frame
                
            def get(self, prop_id):
                if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                    return self.width
                elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                    return self.height
                return 0
                
            def isOpened(self):
                return self.is_opened
                
            def release(self):
                self.is_opened = False
                
            def set(self, prop_id, value):
                return True
        
        return BlankVideoCapture(width, height)
    
    def estimate_distance_with_midas(self, depth_map, box):
        """Estimate distance using MiDaS depth map"""
        if depth_map is None:
            return None
            
        x1, y1, x2, y2 = box
        # Extract the depth region for the object
        object_depth_region = depth_map[y1:y2, x1:x2]
        if object_depth_region.size == 0:
            return None
            
        # Use the median depth value for robustness
        median_depth = np.median(object_depth_region)
        
        # Convert relative depth to approximate meters
        # This is a simplification and would need calibration for accurate values
        depth_meters = median_depth * 10.0  # Scale factor needs calibration
        
        return depth_meters
    
    def estimate_distance(self, box, class_name, depth_map=None):
        """Estimate distance to object using either MiDaS or apparent size"""
        # If we have a depth map, use it for more accurate distance estimation
        if depth_map is not None:
            midas_distance = self.estimate_distance_with_midas(depth_map, box)
            if midas_distance is not None:
                return midas_distance
                
        # Fallback to reference size method
        if class_name not in self.reference_objects:
            return None
            
        # Extract box dimensions
        x1, y1, x2, y2 = box
        object_height_pixels = y2 - y1
        
        # Simple inverse proportion to estimate distance
        reference_height_cm = self.reference_objects[class_name]
        focal_length = 500  # Approximation
        
        distance_cm = (reference_height_cm * focal_length) / object_height_pixels
        return distance_cm / 100  # Convert to meters
    
    def get_direction(self, box):
        """Determine direction of object relative to center of frame"""
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) // 2
        
        # Calculate horizontal position relative to center
        rel_x = box_center_x - self.frame_center_x
        
        # Define direction based on position
        if abs(rel_x) < self.frame_width * 0.1:
            return "directly ahead"
        elif rel_x < -self.frame_width * 0.3:
            return "far to your left"
        elif rel_x < 0:
            return "to your left"
        elif rel_x > self.frame_width * 0.3:
            return "far to your right"
        else:
            return "to your right"

    def plan_safe_path(self, obstacles, depth_map=None):
        """Find a safe path through obstacles using both object detection and depth information"""
        # If no obstacles detected and no depth map, path is clear
        if not obstacles and depth_map is None:
            return "Path is clear ahead"
            
        # Divide the frame into 7 vertical sections for more granular navigation
        section_width = self.frame_width // 7
        sections = [0] * 7  # Danger score for each section
        
        # Score sections based on detected obstacles
        for obj in obstacles:
            x1, y1, x2, y2 = obj['box']
            center_x = (x1 + x2) // 2
            section = min(6, center_x // section_width)
            
            # Weight by distance - closer objects are more dangerous
            distance = obj['distance'] if obj['distance'] else 5.0
            danger = 10.0 / (distance + 0.1)  # Avoid division by zero
            sections[section] += danger
        
        # Use depth information to enhance path planning
        if depth_map is not None:
            # Consider the bottom half of the frame (ground level obstacles)
            lower_region = depth_map[self.frame_height//2:, :]
            
            # Divide into sections matching our planning sections
            for i in range(7):
                start_x = i * section_width
                end_x = (i + 1) * section_width
                section_depth = lower_region[:, start_x:end_x]
                
                # Calculate danger based on proximity of objects
                if section_depth.size > 0:
                    # Find closest points in this section
                    min_depth = np.min(section_depth)
                    
                    # Add danger score based on depth
                    depth_danger = 5.0 / (min_depth * 10.0 + 0.1)  # Convert to meters
                    sections[i] += depth_danger
        
        # Find section with lowest danger score
        min_danger = min(sections)
        best_sections = [i for i, danger in enumerate(sections) if danger == min_danger]
        
        # Choose the most central safe section
        preferred_section = min(best_sections, key=lambda x: abs(x - 3))
        
        # Generate navigation guidance
        if min_danger < 1.0:  # Very safe path
            if preferred_section == 3:
                return "Path is clear directly ahead"
            elif preferred_section < 3:
                shift = "slightly" if preferred_section >= 2 else "more"
                return f"Move {shift} to the left for clearest path"
            else:
                shift = "slightly" if preferred_section <= 4 else "more"
                return f"Move {shift} to the right for clearest path"
        elif min_danger < 3.0:  # Somewhat safe path
            if preferred_section == 3:
                return "Proceed with caution straight ahead"
            elif preferred_section < 3:
                shift = "slightly" if preferred_section >= 2 else "more"
                return f"Move {shift} to the left and proceed with caution"
            else:
                shift = "slightly" if preferred_section <= 4 else "more"
                return f"Move {shift} to the right and proceed with caution"
        else:  # No safe path
            return "Caution! All paths have obstacles. Consider stopping or turning around"
    
    def process_frame(self, frame):
        """Process a single frame with YOLO detection and MiDaS depth estimation"""
        # Load models if not loaded yet (lazy loading)
        if self.lazy_loading and not self.models_loaded:
            self._load_models()
            if not self.models_loaded:
                # Return unprocessed frame if models couldn't be loaded
                return frame, [], []
        
        # Initialize variables
        depth_map = None
        processed_frame = frame.copy()
        detection_data = []
        obstacles = []
        
        # Run YOLO detection
        try:
            results = self.model(frame, conf=self.object_detector.confidence_threshold)
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            cv2.putText(processed_frame, "YOLO Error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame, detection_data, []
        
        # Process depth with MiDaS only occasionally for better performance
        self.skip_depth_count = (self.skip_depth_count + 1) % self.skip_depth_frames
        if self.midas is not None and self.skip_depth_count == 0:
            try:
                # Handle different MiDaS transform formats - try both styles to be compatible
                try:
                    # Get frame dimensions
                    height, width, _ = frame.shape
                    
                    # First try the new dictionary-based transform
                    img_input = self.transform({"image": frame})["image"]
                except (TypeError, KeyError):
                    # If that fails, try the older direct transform
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_input = self.transform(img)
                
                # Move input to device
                img_input = img_input.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    prediction = self.midas(img_input)
                    
                    # Handle different output formats
                    if isinstance(prediction, dict):
                        # Some MiDaS models return a dict
                        prediction = prediction["out"]
                    
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                
                # Get depth map as numpy array
                depth_map = prediction.cpu().numpy()
                
                # Normalize for visualization
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                if depth_max - depth_min > 0:
                    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                else:
                    depth_map = depth_map
                
                # Convert to 8-bit and colormap
                depth_map_normalized = (depth_map * 255).astype(np.uint8)
                self.depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
                
                # Draw the colorized depth map at the top right corner (simplified for performance)
                h, w = processed_frame.shape[:2]
                corner_size = (w//4, h//4)  # 1/4 of the original size
                depth_resized = cv2.resize(self.depth_colormap, corner_size)
                processed_frame[0:corner_size[1], w-corner_size[0]:w] = depth_resized
                
            except Exception as e:
                print(f"Error processing depth: {e}")
                depth_map = None
        
        # Extract detections
        try:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    try:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence
                        confidence = float(box.conf)
                        
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Estimate distance using both methods
                        distance = self.estimate_distance((x1, y1, x2, y2), class_name, depth_map)
                        
                        # Get direction
                        direction = self.get_direction((x1, y1, x2, y2))
                        
                        # Add to detections
                        obj_data = {
                            'class': class_name,
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2),
                            'distance': distance,
                            'direction': direction
                        }
                        detection_data.append(obj_data)
                        
                        # Add to obstacles list if relevant
                        if class_name not in ["wall", "ceiling", "floor", "sky"] and y2 > self.frame_height * 0.5:
                            obstacles.append(obj_data)
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue
        except Exception as e:
            print(f"Error processing YOLO results: {e}")
        
        # Apply object tracking for smoother detection
        try:
            tracked_objects = self.object_detector.track_objects(detection_data)
        except Exception as e:
            print(f"Error in object tracking: {e}")
            tracked_objects = detection_data  # Fallback to untracked detections
        
        # Draw tracked objects on frame
        for obj in tracked_objects:
            try:
                x1, y1, x2, y2 = obj['box']
                class_name = obj['class']
                distance = obj['distance']
                direction = obj['direction']
                
                # Draw rectangle and label on the frame
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display object info with distance
                label = f"{class_name}"
                if distance:
                    label += f" {distance:.1f}m"
                label += f" {direction}"
                
                cv2.putText(processed_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error drawing tracked object: {e}")
                continue
        
        # Update obstacle memory
        if obstacles:
            self.obstacle_memory.append(obstacles)
        
        # Check for hazards using both object detection and depth information
        try:
            hazards = self.hazard_detector.detect_hazards(tracked_objects, depth_map)
        except Exception as e:
            print(f"Error detecting hazards: {e}")
            hazards = []
        
        # Plan safe path considering both obstacles and depth map
        if self.obstacle_memory:
            try:
                all_obstacles = [item for sublist in self.obstacle_memory for item in sublist]
                self.safe_path = self.plan_safe_path(all_obstacles, depth_map)
                
                # Display path guidance on screen
                cv2.putText(processed_frame, self.safe_path, (10, self.frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error planning safe path: {e}")
                self.safe_path = "Error in path planning"
        
        # Update current detections for use by other functions
        self.current_detections = tracked_objects  # Use tracked objects instead of raw detections
        
        # OCR processing if needed
        current_time = time.time()
        if self.ocr_active and current_time - self.last_ocr_time > self.ocr_cooldown:
            text = self.text_detector.perform_ocr(frame)
            if text:
                self.speech_manager.announce(f"Text detected: {text}")
            self.last_ocr_time = current_time
        
        # Prepare speech announcement if needed
        if current_time - self.last_speech_time > self.speech_cooldown:
            announcements = []
            play_proximity_alert = False
            
            # Handle hazard warnings first (priority)
            if hazards:
                for hazard in hazards:
                    if hazard['type'] == 'unmarked_obstacle':
                        warning = f"Warning: Obstacle {hazard['direction']} about {hazard['distance']:.1f} meters away"
                    else:
                        warning = f"Caution: {hazard['object']} {hazard['direction']} about {hazard['distance']:.1f} meters away"
                    announcements.append(warning)
                    
                    # Play audio alert for very close hazards
                    if hazard['distance'] < 1.5:
                        play_proximity_alert = True
            
            # Add path guidance
            if self.safe_path:
                announcements.append(self.safe_path)
                
            # Object announcements
            # Sort objects by distance (closest first)
            sorted_objects = sorted([obj for obj in detection_data if obj['distance']], 
                                   key=lambda x: x['distance'])
            
            # Announce closest objects
            if sorted_objects:
                closest_objects = sorted_objects[:2]  # Limit to 2 closest objects
                
                obj_announcement = "I detect "
                obj_descriptions = []
                
                for obj in closest_objects:
                    desc = f"a {obj['class']} {obj['direction']}"
                    if obj['distance']:
                        desc += f", about {obj['distance']:.1f} meters away"
                    obj_descriptions.append(desc)
                    
                    # Play object detection sound for special objects
                    if obj['class'] in ["person", "car", "traffic light", "dog", "cat"]:
                        self.audio_manager.play_sound("object_detected")
                
                obj_announcement += ", ".join(obj_descriptions)
                announcements.append(obj_announcement)
            
            # Special alerts for important objects
            traffic_lights = [obj for obj in detection_data if obj['class'] == "traffic light"]
            if traffic_lights:
                traffic_light_msg = "Traffic light detected"
                # For directional guidance
                if len(traffic_lights) == 1:
                    tl = traffic_lights[0]
                    traffic_light_msg += f" {tl['direction']}"
                    if tl['distance']:
                        traffic_light_msg += f", about {tl['distance']:.1f} meters away"
                
                announcements.append(traffic_light_msg)
                
            # Play proximity alert sound if needed
            if play_proximity_alert:
                # Play sound first for immediate alert
                self.audio_manager.play_sound("proximity_alert")
                print("Playing proximity alert due to close hazard")
                
            # Make the announcement
            if announcements:
                announcement_text = ". ".join(announcements)
                self.speech_manager.announce(announcement_text)
                print(f"ANNOUNCEMENT: {announcement_text}")
                self.last_speech_time = current_time
        
        # Scene description (less frequent)
        if current_time - self.last_scene_time > self.scene_description_cooldown:
            scene_description = self.environment_detector.generate_scene_description(detection_data, depth_map)
            self.speech_manager.announce(scene_description)
            self.last_scene_time = current_time
        
        # Return the processed frame and detection data
        return processed_frame, tracked_objects, hazards
    
    # Voice command handlers
    def cmd_describe_scene(self):
        """Voice command to describe the current scene"""
        scene_description = self.environment_detector.generate_scene_description(self.current_detections, self.depth_map)
        self.speech_manager.announce(scene_description, priority=True)
        return scene_description
    
    def cmd_identify_objects(self):
        """Voice command to identify objects in the scene"""
        if not self.current_detections:
            self.speech_manager.announce("I don't see any objects clearly right now.", priority=True)
            return "No objects detected"
            
        # Group objects by class
        objects_by_class = {}
        for obj in self.current_detections:
            class_name = obj['class']
            if class_name in objects_by_class:
                objects_by_class[class_name].append(obj)
            else:
                objects_by_class[class_name] = [obj]
                
        # Create announcement
        announcement = "I can see: "
        object_descriptions = []
        
        for class_name, objects in objects_by_class.items():
            count = len(objects)
            desc = f"{count} {class_name}{'s' if count > 1 else ''}"
            object_descriptions.append(desc)
            
        announcement += ", ".join(object_descriptions)
        self.speech_manager.announce(announcement, priority=True)
        return announcement
    
    def cmd_read_text(self):
        """Voice command to read text in the scene"""
        self.speech_manager.announce("Looking for text to read...", priority=True)
        # Set OCR to activate on next frame
        self.ocr_active = True
        self.last_ocr_time = 0
        return "OCR activated"
    
    def cmd_navigate(self):
        """Voice command for navigation assistance"""
        if self.safe_path:
            # Play navigation sound first
            self.audio_manager.play_sound("proximity_alert")
            
            # Then announce guidance
            guidance = f"Navigation guidance: {self.safe_path}"
            self.speech_manager.announce(guidance, priority=True)
            
            # Log success
            print(f"Navigation guidance: {self.safe_path}")
            return self.safe_path
        else:
            # Try to generate a path based on current detections
            sorted_objects = sorted([obj for obj in self.current_detections if obj['distance']], 
                                   key=lambda x: x['distance'])
            
            # If we have any detectable objects
            if sorted_objects:
                # Get closest object for navigation
                closest = sorted_objects[0]
                guidance = f"I detect a {closest['class']} {closest['direction']}. Move to avoid it."
                
                # Play sound first
                self.audio_manager.play_sound("proximity_alert")
                
                # Then announce guidance
                self.speech_manager.announce(guidance, priority=True)
                print(f"Basic navigation: {guidance}")
                return guidance
            else:
                # No objects detected
                message = "I don't have enough information to provide navigation guidance right now."
                self.speech_manager.announce(message, priority=True)
                print(message)
                return "No navigation guidance available"
    
    def cmd_remember_location(self):
        """Voice command to remember the current location"""
        # For a real system, this would use GPS or other positioning
        # Here we're just using a placeholder
        timestamp = time.strftime("%H:%M:%S")
        landmark_name = f"Landmark {len(self.landmarks) + 1}"
        
        # Ask for a name
        self.speech_manager.announce("What would you like to call this location?", priority=True)
        # In a real system, we'd use speech recognition here
        # For now, just use the auto-generated name
        
        self.landmarks[landmark_name] = {
            'time': timestamp,
            'description': self.environment_detector.generate_scene_description(self.current_detections, self.depth_map)
        }
        
        self.speech_manager.announce(f"I've remembered this location as {landmark_name}", priority=True)
        return f"Location saved as {landmark_name}"
    
    def cmd_locate_landmark(self):
        """Voice command to find previously remembered landmarks"""
        if not self.landmarks:
            self.speech_manager.announce("You haven't saved any landmarks yet.", priority=True)
            return "No landmarks saved"
            
        # List available landmarks
        landmark_list = ", ".join(self.landmarks.keys())
        self.speech_manager.announce(f"You have saved these landmarks: {landmark_list}", priority=True)
        return "Landmarks listed"

    def cmd_help(self):
        """Voice command to list available commands"""
        help_text = ("Available commands: "
                    "describe - get a scene description, "
                    "identify - list objects in view, "
                    "read - detect and read text, "
                    "navigate - get navigation guidance, "
                    "remember - save current location, "
                    "locate - find saved landmarks, "
                    "help - list these commands")
        self.speech_manager.announce(help_text, priority=True)
        return help_text
        
    def run(self):
        """Run the system's main loop"""
        prev_frame = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            try:
                # Process the frame with all our detection methods
                processed_frame, detections, hazards = self.process_frame(frame)
                
                # Additional detections from specialized detectors
                # (moved to their respective detector classes)
                
                # Environment recognition (less frequent)
                if time.time() - self.last_scene_time > self.scene_description_cooldown:
                    env_type = self.environment_detector.recognize_environment(self.current_detections)
                    cv2.putText(processed_frame, f"Environment: {env_type}", (10, self.frame_height - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
                
                # Display the processed frame
                cv2.imshow('Enhanced Nayan', processed_frame)
                
                # Return the processed frame for other uses (like web streaming)
                yield processed_frame
                
                # Exit on ESC key
                if cv2.waitKey(1) == 27:
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows() 