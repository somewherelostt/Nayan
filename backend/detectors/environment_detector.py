"""
Environment detector module for Nayan
"""
import numpy as np


class EnvironmentDetector:
    def __init__(self):
        # Environment indicators for scene categorization
        self.indoor_indicators = ["chair", "couch", "bed", "tv", "dining table", "refrigerator", "oven", "sink", "toilet", "desk"]
        self.outdoor_indicators = ["car", "bicycle", "traffic light", "tree", "bench", "bus"]
        
        # Environment type definitions
        self.environments = {
            "kitchen": ["refrigerator", "microwave", "oven", "sink", "bowl", "cup", "bottle", "knife", "spoon", "fork"],
            "living room": ["sofa", "chair", "tv", "remote", "book", "vase", "potted plant"],
            "bathroom": ["toilet", "sink", "toothbrush", "hair drier"],
            "bedroom": ["bed", "chair", "clock"],
            "office": ["chair", "desk", "laptop", "keyboard", "mouse", "monitor"],
            "street": ["car", "truck", "bus", "traffic light", "stop sign", "bicycle", "motorcycle", "person"],
            "store": ["bottle", "person", "chair", "refrigerator", "vase"],
            "park": ["bench", "bicycle", "dog", "person", "potted plant", "tree"]
        }
    
    def generate_scene_description(self, detection_data, depth_data=None):
        """Generate a comprehensive description of the scene"""
        if not detection_data and depth_data is None:
            return "I don't see any objects clearly in the scene"
            
        # Count object types
        object_counts = {}
        for obj in detection_data:
            class_name = obj['class']
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1
        
        # Categorize the scene
        indoor_score = sum(object_counts.get(item, 0) for item in self.indoor_indicators)
        outdoor_score = sum(object_counts.get(item, 0) for item in self.outdoor_indicators)
        
        scene_type = "indoors" if indoor_score > outdoor_score else "outdoors"
        
        # Count people and obstacles
        people_count = object_counts.get("person", 0)
        obstacle_count = sum(object_counts.values()) - people_count
        
        # Generate description
        description = f"You appear to be {scene_type}. "
        
        # Add depth context if available
        if depth_data is not None:
            # Calculate average depth in different regions
            height = depth_data.shape[0]
            top_half = depth_data[:height//2, :]
            bottom_half = depth_data[height//2:, :]
            
            avg_distance_ahead = np.median(depth_data) * 10.0  # Convert to meters
            
            if avg_distance_ahead < 1.5:
                description += "You're in a confined space. "
            elif avg_distance_ahead < 3.0:
                description += "You're in a medium-sized space. "
            else:
                description += "You're in an open area. "
            
            # Detect walls or large surfaces
            if np.std(top_half) < 0.1 and np.median(top_half) < 0.5:
                description += "There might be a wall close in front of you. "
            
            if np.std(bottom_half) < 0.1:
                if np.median(bottom_half) < 0.2:
                    description += "The floor appears very close, you might be looking down. "
        
        if people_count > 0:
            description += f"There {'is' if people_count == 1 else 'are'} {people_count} {'person' if people_count == 1 else 'people'} around you. "
        
        if obstacle_count > 0:
            description += f"I can see {obstacle_count} objects that could be obstacles. "
            
        # Add details about prominent objects
        prominent_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        if prominent_objects:
            objects_text = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" for name, count in prominent_objects])
            description += f"The main objects I detect are {objects_text}."
            
        return description
    
    def recognize_environment(self, detection_data):
        """Identify the type of environment based on detected objects"""
        if not detection_data:
            return "Unknown environment"
            
        # Count occurrences of each environment's indicators
        env_scores = {env: 0 for env in self.environments}
        
        # Get all detected classes
        detected_classes = [obj['class'] for obj in detection_data]
        
        # Score each environment
        for env, indicators in self.environments.items():
            for indicator in indicators:
                env_scores[env] += detected_classes.count(indicator)
        
        # Find environment with highest score
        if max(env_scores.values()) > 0:
            likely_env = max(env_scores, key=env_scores.get)
            confidence = min(100, env_scores[likely_env] * 20)  # Simple confidence score
            return f"{likely_env} ({confidence}% confidence)"
        else:
            return "Unknown environment"
    
    def analyze_crowd_density(self, detection_data):
        """Estimate the density of people in a scene"""
        people = [obj for obj in detection_data if obj['class'] == 'person']
        
        if not people:
            return "No people detected"
            
        # Count the number of people
        count = len(people)
        
        # Calculate average distance
        distances = [p['distance'] for p in people if p['distance']]
        if distances:
            avg_distance = sum(distances) / len(distances)
        else:
            avg_distance = None
            
        # Calculate density based on count
        if count <= 2:
            density = "few people"
        elif count <= 5:
            density = "several people"
        elif count <= 10:
            density = "moderately crowded"
        else:
            density = "very crowded"
            
        if avg_distance is not None:
            proximity = f", nearest about {min(distances):.1f} meters away" if distances else ""
        else:
            proximity = ""
            
        return f"{count} people detected, {density}{proximity}"
    
    def map_surroundings(self, detection_data, frame_width, frame_height):
        """Create a simplified spatial map of surroundings"""
        if not detection_data:
            return "Insufficient data to map surroundings"
            
        # Divide the space into 9 sectors (3x3 grid)
        sectors = [[[] for _ in range(3)] for _ in range(3)]
        
        # Assign objects to sectors
        for obj in detection_data:
            box = obj['box']
            x1, y1, x2, y2 = box
            
            # Find center of the object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Determine sector
            sector_x = min(2, center_x * 3 // frame_width)
            sector_y = min(2, center_y * 3 // frame_height)
            
            # Add to appropriate sector
            sectors[sector_y][sector_x].append(obj)
        
        # Create a textual map
        sector_names = [
            ["top left", "top center", "top right"],
            ["left", "center", "right"],
            ["bottom left", "bottom center", "bottom right"]
        ]
        
        # Generate map description
        description = "Spatial map: "
        for y in range(3):
            for x in range(3):
                if sectors[y][x]:
                    # Count objects by type
                    obj_counts = {}
                    for obj in sectors[y][x]:
                        class_name = obj['class']
                        if class_name in obj_counts:
                            obj_counts[class_name] += 1
                        else:
                            obj_counts[class_name] = 1
                    
                    # Create description for this sector
                    items = []
                    for class_name, count in obj_counts.items():
                        items.append(f"{count} {class_name}{'s' if count > 1 else ''}")
                    
                    sector_desc = f"In {sector_names[y][x]}: {', '.join(items)}. "
                    description += sector_desc
        
        return description 