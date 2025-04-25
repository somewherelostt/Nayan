"""
Object detector module for Nayan
"""
import cv2
import numpy as np
import time


class ObjectDetector:
    def __init__(self):
        # Reference sizes for distance estimation (in cm)
        self.reference_objects = {
            "person": 170,  # Average height in cm
            "car": 150,     # Average height in cm
            "bottle": 20,   # Average height in cm
            "chair": 80,    # Average height in cm
            "dog": 50,      # Average height in cm
            "cat": 30,      # Average height in cm
            "table": 75,    # Average height in cm
            "laptop": 25,   # Average height in cm
            "cell phone": 15,  # Average height in cm
            "book": 25,     # Average height in cm
            "backpack": 45  # Average height in cm
        }
        
        # Default confidence threshold for detections
        self.confidence_threshold = 0.45
        
        # Tracking parameters for smoother detection
        self.last_detections = {}  # Store previous detections for tracking
        self.detection_history = {}  # Store detection history for each object
        self.max_tracking_age = 5  # Frames to track an object after it disappears
        self.min_detection_count = 2  # Minimum detections to consider object real
        self.last_process_time = time.time()
        
        # Movement detection parameters
        self.movement_threshold = 20  # Minimum pixel difference for movement
        self.movement_area_threshold = 400  # Minimum area to consider significant movement
        self.movement_cooldown = 0.5  # Seconds between movement detections
        self.last_movement_time = 0
        
        # Initialize GPU-optimized background subtractor if available
        try:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        except:
            self.bg_subtractor = None
            print("Warning: Could not initialize background subtractor")
        
    def detect_crosswalk(self, frame):
        """Detect crosswalk patterns in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to improve detection
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for patterns of parallel lines
        large_rect_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size and aspect ratio
            if w > 20 and h > 20 and 0.1 < w/h < 10:
                large_rect_contours.append(contour)
        
        # Count potential crosswalk lines
        if len(large_rect_contours) >= 3:
            # Analyze the pattern
            bounding_rects = [cv2.boundingRect(c) for c in large_rect_contours]
            
            # Sort by y-coordinate
            bounding_rects.sort(key=lambda r: r[1])
            
            # Check for roughly parallel lines
            parallel_count = 0
            for i in range(len(bounding_rects) - 1):
                r1 = bounding_rects[i]
                r2 = bounding_rects[i+1]
                
                # Check if heights are similar and they're horizontally aligned
                if abs(r1[3] - r2[3]) < 15 and abs(r1[1] - r2[1]) < 50:
                    parallel_count += 1
            
            if parallel_count >= 2:
                return True, bounding_rects
        
        return False, []
    
    def detect_traffic_signals(self, frame):
        """Detect traffic signal colors"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for traffic light colors
        red_lower1 = np.array([0, 120, 120])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 120, 120])
        red_upper2 = np.array([180, 255, 255])
        
        green_lower = np.array([40, 70, 70])
        green_upper = np.array([90, 255, 255])
        
        yellow_lower = np.array([15, 150, 150])
        yellow_upper = np.array([35, 255, 255])
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Count non-zero pixels in each mask
        red_count = cv2.countNonZero(red_mask)
        green_count = cv2.countNonZero(green_mask)
        yellow_count = cv2.countNonZero(yellow_mask)
        
        # Determine dominant color
        color_counts = {'red': red_count, 'green': green_count, 'yellow': yellow_count}
        dominant_color = max(color_counts, key=color_counts.get)
        
        # Only consider it a traffic signal if there's a significant amount of color
        threshold = 400  # Reduced threshold for better detection
        if color_counts[dominant_color] > threshold:
            return True, dominant_color
        
        return False, None
    
    def track_objects(self, current_detections):
        """Track objects between frames for more stable detection"""
        current_time = time.time()
        
        # Performance optimization - limit processing time
        elapsed = current_time - self.last_process_time
        if elapsed < 0.01:  # Avoid processing too frequently
            return current_detections
        
        self.last_process_time = current_time
        
        # Update detection history with current detections
        current_tracked = {}
        
        # Match current detections with existing tracked objects - optimized for speed
        for obj in current_detections:
            obj_id = f"{obj['class']}_{len(current_tracked)}"
            best_match = None
            best_iou = 0.3  # Minimum IOU threshold
            
            # Find best match in previous detections - optimized for fewer comparisons
            for prev_id, prev_obj in self.last_detections.items():
                if prev_obj['class'] == obj['class']:
                    # Quick check for approximate position before calculating IoU
                    px1, py1, px2, py2 = prev_obj['box']
                    cx1, cy1, cx2, cy2 = obj['box']
                    
                    # Check if centers are close enough
                    prev_center_x = (px1 + px2) / 2
                    prev_center_y = (py1 + py2) / 2
                    curr_center_x = (cx1 + cx2) / 2
                    curr_center_y = (cy1 + cy2) / 2
                    
                    # Quick distance check before expensive IoU calculation
                    dist = ((prev_center_x - curr_center_x)**2 + (prev_center_y - curr_center_y)**2) ** 0.5
                    max_dist = max((px2 - px1), (py2 - py1)) * 0.75  # Allow movement up to 75% of object size
                    
                    if dist <= max_dist:
                        iou = self._calculate_iou(obj['box'], prev_obj['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = prev_id
            
            # If match found, use the same ID
            if best_match:
                obj_id = best_match
                # Update the detection count
                if obj_id in self.detection_history:
                    self.detection_history[obj_id]['count'] += 1
                    self.detection_history[obj_id]['last_seen'] = current_time
                    
                    # Update position smoothly for better visual tracking
                    if 'box' in self.detection_history[obj_id]:
                        prev_box = self.detection_history[obj_id]['box']
                        curr_box = obj['box']
                        # Apply smoothing (80% new position, 20% old position)
                        smoothed_box = [
                            int(0.8 * curr_box[0] + 0.2 * prev_box[0]),
                            int(0.8 * curr_box[1] + 0.2 * prev_box[1]),
                            int(0.8 * curr_box[2] + 0.2 * prev_box[2]),
                            int(0.8 * curr_box[3] + 0.2 * prev_box[3])
                        ]
                        obj['box'] = tuple(smoothed_box)
                    
                    # Store current box for next frame
                    self.detection_history[obj_id]['box'] = obj['box']
                else:
                    self.detection_history[obj_id] = {
                        'count': 1,
                        'last_seen': current_time,
                        'box': obj['box']
                    }
            else:
                # New object
                self.detection_history[obj_id] = {
                    'count': 1,
                    'last_seen': current_time,
                    'box': obj['box']
                }
            
            # Add to current tracked objects
            current_tracked[obj_id] = obj
        
        # Add objects from previous frame that weren't matched but are still being tracked
        for prev_id, prev_obj in self.last_detections.items():
            if prev_id not in current_tracked:
                history = self.detection_history.get(prev_id, {'count': 0, 'last_seen': 0})
                
                # If object has been detected enough times and hasn't been gone too long
                if (history['count'] >= self.min_detection_count and 
                    current_time - history['last_seen'] < self.max_tracking_age):
                    # Add to current tracked objects
                    current_tracked[prev_id] = prev_obj
                    # Update last seen to mark as tracked, not directly detected
                    self.detection_history[prev_id]['tracked'] = True
        
        # Update last detections for next frame
        self.last_detections = current_tracked
        
        # Sort objects by distance for better display
        tracked_list = list(current_tracked.values())
        # Try to sort by distance if available
        try:
            tracked_list.sort(key=lambda x: float('inf') if x.get('distance') is None else x.get('distance'))
        except:
            pass  # Skip sorting if there's an error
        
        # Return the list of tracked objects
        return tracked_list
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
        
    def detect_human_pose(self, frame, current_detections):
        """Detect human poses to recognize gestures or interactions"""
        try:
            # Check if there are people detected by YOLO
            people = [obj for obj in current_detections if obj['class'] == 'person']
            
            if people:
                closest_person = min(people, key=lambda p: p['distance'] if p['distance'] else float('inf'))
                
                # Calculate if they might be facing the user
                box = closest_person['box']
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Use aspect ratio as a crude approximation of orientation
                aspect_ratio = width / height if height > 0 else 0
                
                # More sophisticated approach for pose detection
                if width > 100 and height > 200:  # Only try to detect pose for larger people
                    # Extract person ROI for better analysis
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size > 0:
                        # Calculate horizontal symmetry as additional pose indicator
                        if person_roi.shape[1] > 1:
                            left_half = person_roi[:, :person_roi.shape[1]//2]
                            right_half = cv2.flip(person_roi[:, person_roi.shape[1]//2:], 1)
                            
                            # Resize if needed
                            if left_half.shape != right_half.shape:
                                min_width = min(left_half.shape[1], right_half.shape[1])
                                left_half = left_half[:, :min_width]
                                right_half = right_half[:, :min_width]
                            
                            # Calculate symmetry score
                            if left_half.size > 0 and left_half.shape == right_half.shape:
                                diff = cv2.absdiff(left_half, right_half)
                                symmetry_score = np.mean(diff)
                                
                                # Lower score means more symmetric (facing camera)
                                if symmetry_score < 50:
                                    pose = "facing you"
                                else:
                                    pose = "sideways"
                            else:
                                pose = "unknown pose"
                        else:
                            pose = "unknown pose"
                    else:
                        # Fallback to aspect ratio if ROI extraction failed
                        if aspect_ratio > 0.5:  # Wider than tall - might be facing sideways
                            pose = "sideways"
                        else:
                            pose = "facing you"
                else:
                    # Fallback for small detections
                    if aspect_ratio > 0.5:
                        pose = "sideways"
                    else:
                        pose = "facing you"
                
                return True, closest_person['distance'], pose
            
            return False, None, None
            
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return False, None, None
            
    def detect_movement(self, prev_frame, curr_frame):
        """Detect movement in the scene between frames"""
        current_time = time.time()
        
        # Check cooldown to avoid too frequent detections
        if current_time - self.last_movement_time < self.movement_cooldown:
            return False, None
            
        if prev_frame is None or curr_frame is None:
            return False, None
            
        # Try to use background subtractor if available (faster)
        if self.bg_subtractor is not None:
            try:
                # Apply background subtraction
                fg_mask = self.bg_subtractor.apply(curr_frame)
                
                # Apply threshold to get binary mask
                _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                
                # Apply morphological operations to reduce noise
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter small contours
                significant_contours = [c for c in contours if cv2.contourArea(c) > self.movement_area_threshold]
                
                if significant_contours:
                    # Determine where the most movement is occurring
                    max_area_contour = max(significant_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(max_area_contour)
                    
                    # Get frame dimensions and location
                    movement_location = self._get_movement_location(curr_frame, x, y, w, h)
                    
                    # Update last movement time
                    self.last_movement_time = current_time
                    
                    return True, movement_location
                    
                return False, None
                
            except Exception as e:
                print(f"Error in background subtraction: {e}")
                # Fall back to frame difference method
        
        # Fallback: Use frame difference method
        try:
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Apply thresholding
            _, thresh = cv2.threshold(diff, self.movement_threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours
            significant_contours = [c for c in contours if cv2.contourArea(c) > self.movement_area_threshold]
            
            if significant_contours:
                # Determine where the most movement is occurring
                max_area_contour = max(significant_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_area_contour)
                
                # Get movement location
                movement_location = self._get_movement_location(curr_frame, x, y, w, h)
                
                # Update last movement time
                self.last_movement_time = current_time
                
                return True, movement_location
            
            return False, None
            
        except Exception as e:
            print(f"Error in movement detection: {e}")
            return False, None
    
    def _get_movement_location(self, frame, x, y, w, h):
        """Determine movement location relative to frame center"""
        # Calculate center of movement
        center_x = x + w//2
        center_y = y + h//2
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Determine position relative to frame center
        if center_x < width * 0.33:
            horizontal = "left"
        elif center_x > width * 0.66:
            horizontal = "right"
        else:
            horizontal = "center"
            
        if center_y < height * 0.33:
            vertical = "top"
        elif center_y > height * 0.66:
            vertical = "bottom"
        else:
            vertical = "middle"
            
        return f"{vertical} {horizontal}" 