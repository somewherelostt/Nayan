"""
Model loader for Nayan
"""
import os
import torch
from ultralytics import YOLO


class ModelLoader:
    def __init__(self):
        # Get the device
        self.device = self.get_device()
        
        # Set model paths
        self.yolo_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'yolov8n.pt')
        self.midas_model_type = 'MiDaS_small'  # Options: MiDaS_small, DPT_Large, DPT_Hybrid
        
        # Force MiDaS to CPU if not enough VRAM or for compatibility
        self.force_midas_cpu = True
        
        # Loaded models
        self.yolo_model = None
        self.midas_model = None
        self.midas_transform = None
    
    def get_device(self):
        """Get the best available device for model inference"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # For Apple M1/M2 GPUs
        else:
            return torch.device("cpu")
    
    def load_yolo_model(self):
        """Load YOLO model from PyTorch Hub or local file"""
        try:
            # Check if local model file exists
            if os.path.exists(self.yolo_model_path):
                print(f"Loading YOLO model from {self.yolo_model_path}")
                model = YOLO(self.yolo_model_path)
            else:
                # Download from ultralytics
                print("Loading YOLO model from ultralytics")
                model = YOLO('yolov8n.pt')
            
            self.yolo_model = model
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Trying to load a smaller model...")
            try:
                # Try loading smaller model as fallback
                model = YOLO('yolov8n.pt')
                self.yolo_model = model
                return model
            except Exception as e2:
                print(f"Failed to load YOLO model: {e2}")
                return None
    
    def load_midas_model(self):
        """Load MiDaS depth estimation model"""
        try:
            # Ensure torch is installed
            import torch

            # Import MiDaS specific modules
            try:
                # Try different import styles for compatibility
                try:
                    # New import style
                    from midas.model_loader import default_models, load_model
                    model_path = default_models[self.midas_model_type]
                    midas = load_model(model_path, self.device)
                except (ImportError, ModuleNotFoundError):
                    # Old import style
                    print("Using older MiDaS import style")
                    import cv2
                    midas = cv2.dnn.readNetFromONNX("models/midas_v21_small_256.onnx")
            except (ImportError, ModuleNotFoundError):
                # Direct loading from torch hub
                print("Using torch hub for MiDaS")
                midas = torch.hub.load("intel-isl/MiDaS", self.midas_model_type)
                
                # Move to device (unless forced to CPU)
                device = torch.device("cpu") if self.force_midas_cpu else self.device
                midas.to(device)
                midas.eval()
            
            self.midas_model = midas
            return midas
            
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            print("Depth estimation will not be available")
            return None
    
    def get_midas_transform(self):
        """Get the appropriate transform for the MiDaS model"""
        try:
            # Try to load the transform
            try:
                # Try different import styles for compatibility
                try:
                    # New import style
                    from midas.transforms import Resize, NormalizeImage, PrepareForNet
                    import torchvision.transforms as T
                    
                    # Define transformations
                    transforms = T.Compose([
                        Resize(
                            384, 384,
                            resize_target=None,
                            keep_aspect_ratio=True,
                            ensure_multiple_of=32,
                            resize_method="upper_bound",
                            image_interpolation_method=cv2.INTER_CUBIC,
                        ),
                        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        PrepareForNet(),
                    ])
                    self.midas_transform = transforms
                    return transforms
                    
                except (ImportError, ModuleNotFoundError, NameError):
                    # Old import style or direct Torch Hub loading
                    import torch
                    
                    # Initialize a default transform for Torch Hub MiDaS models
                    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                    
                    if self.midas_model_type == "DPT_Large" or self.midas_model_type == "DPT_Hybrid":
                        transform = transforms.dpt_transform
                    else:
                        transform = transforms.small_transform
                    
                    self.midas_transform = transform
                    return transform
                    
            except Exception as inner_error:
                print(f"Error loading MiDaS transform: {inner_error}")
                print("Creating a basic transform as fallback")
                
                # Create a very basic transform as fallback
                import torch
                import torchvision.transforms as transforms
                
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                self.midas_transform = transform
                return transform
                
        except Exception as e:
            print(f"Error setting up MiDaS transform: {e}")
            return None 