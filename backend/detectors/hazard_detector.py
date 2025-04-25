"""
Hazard detector module for Nayan
"""
import numpy as np


class HazardDetector:
    def __init__(self):
        # Define hazard types
        self.hazard_types = {
            "stairs": ["stairs", "staircase", "step"],
            "water": ["puddle", "pool", "water"],
            "traffic": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "sharp": ["knife", "scissors", "glass"]
        }
        
        # Define risk levels for objects
        self.high_risk_objects = ["car", "bicycle", "motorcycle", "truck", "bus", "fire hydrant", "bench"]
        self.medium_risk_objects = ["chair", "table", "pole", "traffic light", "potted plant"]
        self.low_risk_objects = ["backpack", "umbrella", "handbag"]
    
    def detect_hazards(self, detection_data, depth_map):
        """Identify potential hazards in the environment"""
        hazards = []
        
        # Check for known hazardous objects
        for obj in detection_data:
            class_name = obj['class']
            
            # Check if the object belongs to any hazard category
            for hazard_type, keywords in self.hazard_types.items():
                if any(keyword in class_name for keyword in keywords):
                    distance = obj['distance']
                    if distance and distance < 3.0:  # Only alert for close hazards
                        hazards.append({
                            'type': hazard_type,
                            'object': class_name,
                            'distance': distance,
                            'direction': obj['direction']
                        })
        
        # Use depth map to detect unmarked obstacles (like walls, furniture without labels)
        if depth_map is not None:
            # Look at the lower half of the depth map (ground obstacles)
            height = depth_map.shape[0]
            lower_half = depth_map[height//2:, :]
            
            # Find regions that are very close
            depth_threshold = 0.2  # Threshold in normalized depth units
            close_mask = lower_half < depth_threshold
            
            if np.any(close_mask):
                # Identify the direction of the closest point
                close_points = np.where(close_mask)
                if len(close_points[0]) > 0:
                    # Find the closest point
                    closest_idx = np.argmin(lower_half[close_mask])
                    y, x = close_points[0][closest_idx], close_points[1][closest_idx]
                    
                    # Adjust y coordinate to account for the lower half
                    y += height // 2
                    
                    # Calculate direction
                    width = depth_map.shape[1]
                    if x < width * 0.33:
                        direction = "to your left"
                    elif x > width * 0.66:
                        direction = "to your right"
                    else:
                        direction = "directly ahead"
                    
                    # Add to hazards
                    hazards.append({
                        'type': 'unmarked_obstacle',
                        'object': 'obstacle',
                        'distance': lower_half[y-height//2, x] * 10.0,  # Convert to meters
                        'direction': direction
                    })
        
        return hazards
    
    def detect_obstacle_type(self, obj_data, depth_map):
        """Classify obstacles by type and severity"""
        class_name = obj_data['class']
        distance = obj_data['distance']
        
        if not distance:
            return "unknown", 0
            
        # Default severity based on distance
        if distance < 1.0:
            base_severity = 3  # High
        elif distance < 2.5:
            base_severity = 2  # Medium
        else:
            base_severity = 1  # Low
            
        # Adjust severity based on object type
        if class_name in self.high_risk_objects:
            obstacle_type = "dangerous obstacle"
            severity = base_severity + 1
        elif class_name in self.medium_risk_objects:
            obstacle_type = "fixed obstacle"
            severity = base_severity
        elif class_name in self.low_risk_objects:
            obstacle_type = "small obstacle"
            severity = base_severity - 1
        else:
            obstacle_type = "obstacle"
            severity = base_severity
            
        # Cap severity between 1-3
        severity = max(1, min(3, severity))
        
        return obstacle_type, severity
    
    def detect_stairs(self, frame, depth_map):
        """Detect stairs using depth map patterns"""
        if depth_map is None:
            return False
            
        # Focus on the lower half of the image where stairs would typically be
        lower_half = depth_map[depth_map.shape[0]//2:, :]
        
        # Look for horizontal bands of similar depth
        # First, apply a median filter to smooth the depth map
        try:
            import cv2
            smoothed = cv2.medianBlur(np.float32(lower_half), 5)
            
            # Calculate vertical gradient
            gradient_y = np.diff(smoothed, axis=0)
            
            # Look for significant vertical changes
            threshold = 0.05  # Threshold for significant depth change
            significant_changes = np.abs(gradient_y) > threshold
            
            # Count rows with significant changes
            change_counts = np.sum(significant_changes, axis=1)
            
            # Look for rows with many changes (potential stair edges)
            width = depth_map.shape[1]
            potential_edges = change_counts > (width // 4)
            
            # Check if we have multiple potential edges spaced appropriately
            edge_indices = np.where(potential_edges)[0]
            
            if len(edge_indices) >= 3:
                # Check if edges are regularly spaced (like stairs would be)
                spacing = np.diff(edge_indices)
                avg_spacing = np.mean(spacing)
                std_spacing = np.std(spacing)
                
                # Regular spacing would have low standard deviation
                if std_spacing < (avg_spacing * 0.3) and 5 < avg_spacing < 30:
                    return True
        except Exception as e:
            print(f"Error in stairs detection: {e}")
        
        return False 