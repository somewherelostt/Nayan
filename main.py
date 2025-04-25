"""
Main entry point for the Nayan application
"""
import os
import sys
import cv2
import time
import argparse
import threading
import numpy as np
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.nayan import EnhancedNayan

# Global variables
# Frame rate limiter (fps) - Optimized for real-time performance
FPS = 25
# Frame resize factor (1.0 = original size) - Balanced for performance
RESIZE_FACTOR = 0.8
# Global Nayan instance
nayan = None
# Lightweight mode (for better performance)
LIGHTWEIGHT_MODE = True
# Skip frames (process every N frames) - Balanced for performance
PROCESS_EVERY_N_FRAMES = 3
# Current frame counter
frame_counter = 0
# Last processed detection data
last_detections = []
last_hazards = []
last_safe_path = None
# Debug mode
DEBUG = True
# Camera buffer size - set to 2 for smoother video
CAMERA_BUFFER_SIZE = 2
# JPEG quality for streaming - balanced for performance
JPEG_QUALITY = 90
# Camera availability list - for faster camera detection
available_cameras = []
# Frame dropping threshold (ms)
FRAME_DROP_THRESHOLD = 0.05
# Hardware acceleration flag
USE_HW_ACCEL = True  # Enable for better performance
# Detection confidence threshold
DETECTION_CONFIDENCE = 0.45  # Lower threshold for more detections
# Video codec settings
VIDEO_CODEC = 'MJPG'  # Use Motion JPEG for better quality
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
# Frame interpolation settings
INTERPOLATE_FRAMES = True
MAX_INTERPOLATION_FRAMES = 2
# Color correction settings
COLOR_CORRECTION = True
SATURATION_BOOST = 1.3
CONTRAST_BOOST = 1.2
BRIGHTNESS_BOOST = 10

# Define a function to create a virtual camera if no real camera is available
def create_virtual_camera():
    """Create a virtual camera with sample content"""
    class VirtualCamera:
        def __init__(self):
            self.frame_count = 0
            self.width = 640
            self.height = 480
            
        def read(self):
            # Create a colored frame with moving patterns
            self.frame_count += 1
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Add a moving circle
            radius = 50
            x = int(self.width/2 + self.width/4 * np.sin(self.frame_count * 0.05))
            y = int(self.height/2 + self.height/4 * np.cos(self.frame_count * 0.05))
            cv2.circle(frame, (x, y), radius, (0, 165, 255), -1)
            
            # Add some rectangles
            cv2.rectangle(frame, (20, 20), (100, 100), (0, 255, 0), -1)
            cv2.rectangle(frame, (self.width-120, self.height-120), 
                         (self.width-20, self.height-20), (255, 0, 0), -1)
            
            # Add text
            cv2.putText(frame, "Nayan Virtual Camera", (self.width//4, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, self.height-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return True, frame
            
        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return self.width
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.height
            elif prop_id == cv2.CAP_PROP_FPS:
                return 30
            elif prop_id == cv2.CAP_PROP_BUFFERSIZE:
                return 1
            return 0
                
        def isOpened(self):
            return True
                
        def release(self):
            pass
        
        def set(self, prop_id, value):
            return True
            
    return VirtualCamera()

# Function to detect available cameras
def detect_available_cameras(max_cameras=10):
    """Detect which camera indices are available"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
                print(f"Found working camera at index {i}")
            cap.release()
    return available

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join('frontend', 'templates'),
            static_folder=os.path.join('frontend', 'static'))
# Enable CORS for all routes and origins
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

def configure_camera(cap):
    """Configure camera properties for better performance"""
    if isinstance(cap, cv2.VideoCapture):
        # Set buffer size for smoother video
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        
        # Try to set optimal resolution for quality and performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Try to set optimal FPS for smoother video
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Enable auto focus and auto exposure for better image quality
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Set video codec for better quality
        cap.set(cv2.CAP_PROP_FOURCC, VIDEO_FOURCC)
        
        # Try to enable hardware acceleration if available
        if USE_HW_ACCEL:
            try:
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
            except:
                print("Hardware acceleration not supported")
        
        # Try to improve camera settings for better image quality
        # Brightness (adjust if needed, 0-100)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 55)
        # Contrast (adjust if needed, 0-100)
        cap.set(cv2.CAP_PROP_CONTRAST, 55)
        # Saturation (adjust if needed, 0-100)
        cap.set(cv2.CAP_PROP_SATURATION, 65)
        
        print(f"Camera configured: " + 
              f"Resolution {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, " +
              f"FPS {int(cap.get(cv2.CAP_PROP_FPS))}, " +
              f"Codec: {VIDEO_CODEC}")
    return cap

def get_nayan_instance():
    """Get or create the Nayan instance"""
    global nayan, available_cameras, DETECTION_CONFIDENCE
    if nayan is None:
        try:
            # First try with the specified camera index if it's in the available list
            camera_index = args.camera_index
            if not available_cameras:
                available_cameras = detect_available_cameras()
                
            # If specified camera isn't in available list but others are, use the first available
            if camera_index not in available_cameras and available_cameras:
                camera_index = available_cameras[0]
                print(f"Using available camera at index {camera_index} instead of requested {args.camera_index}")
            
            # Initialize with error handling
            try:
                print(f"Initializing Nayan with camera index {camera_index}")
                nayan = EnhancedNayan(camera_index=camera_index, use_sample_video=args.use_sample_video)
                
                # Update detection confidence if applicable
                if hasattr(nayan, 'object_detector') and hasattr(nayan.object_detector, 'confidence_threshold'):
                    nayan.object_detector.confidence_threshold = DETECTION_CONFIDENCE
                    print(f"Set detection confidence threshold to {DETECTION_CONFIDENCE}")
            except Exception as e:
                print(f"Error initializing Nayan: {e}")
                # Try with virtual camera as fallback
                nayan = EnhancedNayan(camera_index=0, use_sample_video=True) 
                nayan.cap = create_virtual_camera()
            
            # Configure camera for better performance
            if isinstance(nayan.cap, cv2.VideoCapture):
                # Force camera release and reopen to ensure a clean start
                try:
                    cam_index = camera_index
                    nayan.cap.release()
                    time.sleep(0.5)
                    nayan.cap = cv2.VideoCapture(cam_index)
                    nayan.cap = configure_camera(nayan.cap)
                    
                    # Try to read a test frame to ensure camera is working
                    for _ in range(5):  # Try up to 5 times
                        ret, frame = nayan.cap.read()
                        if ret and frame is not None:
                            print("Camera successfully initialized with test frame")
                            # Apply image enhancement to check quality
                            enhanced = enhance_image(frame)
                            if enhanced is not None:
                                print("Image enhancement working properly")
                            break
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Error configuring camera: {e}")
                
            # Check if camera actually opened or is using the blank video
            if isinstance(nayan.cap, cv2.VideoCapture) and not nayan.cap.isOpened():
                print("Failed to open camera, using virtual camera instead")
                nayan.cap = create_virtual_camera()
                
            # Set up announcement forwarder
            setup_announcement_forwarder(nayan)
        except Exception as e:
            print(f"Error initializing Nayan: {e}")
            # Try with virtual camera as fallback
            nayan = EnhancedNayan(camera_index=0, use_sample_video=True) 
            nayan.cap = create_virtual_camera()
            setup_announcement_forwarder(nayan)
    return nayan

def setup_announcement_forwarder(nayan_instance):
    """Set up a forwarder to send announcements to the frontend via WebSocket"""
    original_announce = nayan_instance.speech_manager.announce
    
    def announce_with_websocket(text, priority=False):
        # Call the original method
        result = original_announce(text, priority)
        
        # Forward to WebSocket
        if text:
            print(f"Forwarding announcement to WebSocket: {text}")
            socketio.emit('announcement', {'text': text, 'priority': priority})
        
        return result
    
    # Replace the original method with our modified version
    nayan_instance.speech_manager.announce = announce_with_websocket
    
    # Also set up audio hooks
    original_play_sound = nayan_instance.audio_manager.play_sound
    
    def play_sound_with_websocket(sound_name):
        # Call the original method
        result = original_play_sound(sound_name)
        
        # Forward to WebSocket
        print(f"Forwarding sound to WebSocket: {sound_name}")
        socketio.emit('sound', {'name': sound_name})
        
        return result
    
    # Replace the original method with our modified version
    nayan_instance.audio_manager.play_sound = play_sound_with_websocket

def generate_frames():
    """Generate video frames for streaming"""
    if DEBUG:
        print("Starting video frame generation...")
    
    instance = get_nayan_instance()
    
    # Keep track of the previous frame for motion detection and interpolation
    prev_frame = None
    last_frame_time = time.time()
    frame_count = 0
    error_count = 0
    dropped_frames = 0
    last_successful_frame = None
    consecutive_identical_frames = 0
    interpolation_buffer = []
    consecutive_errors = 0
    
    # Frame counter for skipping frames
    global frame_counter, last_detections, last_hazards, last_safe_path
    
    while True:
        try:
            frame_count += 1
            
            # Apply frame rate limiting but don't sleep too much to ensure continuous frame capture
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Drop frames if processing is too slow
            if elapsed > FRAME_DROP_THRESHOLD:
                dropped_frames += 1
                if DEBUG and dropped_frames % 100 == 0:
                    print(f"Dropped {dropped_frames} frames due to slow processing")
            
            # Apply minimal sleep to prevent CPU overuse but ensure frame capture
            sleep_time = max(0, 1.0/FPS - elapsed)
            if sleep_time > 0 and sleep_time < 0.05:  # Only sleep if it's a short time
                time.sleep(sleep_time)
            
            # Increment frame counter
            frame_counter = (frame_counter + 1) % PROCESS_EVERY_N_FRAMES
            process_this_frame = frame_counter == 0 or not LIGHTWEIGHT_MODE
            
            last_frame_time = time.time()
            
            # Get camera frame
            ret, frame = instance.cap.read()
            
            if not ret or frame is None:
                error_count += 1
                consecutive_errors += 1
                if DEBUG:
                    print(f"Failed to grab frame, attempt {error_count}/3")
                
                # If we have a previous successful frame, use it temporarily
                if last_successful_frame is not None and consecutive_errors < 5:
                    frame = last_successful_frame.copy()
                    ret = True
                    # Add a visual indicator that this is a repeated frame
                    cv2.putText(frame, "Camera Lag", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif error_count >= 3:
                    # Reset the camera if we failed multiple times
                    if DEBUG:
                        print("Resetting camera connection...")
                    try:
                        instance.cap.release()
                        time.sleep(0.5)  # Reduced wait time
                        
                        # Try to find an available camera again
                        global available_cameras
                        available_cameras = detect_available_cameras()
                        
                        if available_cameras:
                            instance.cap = cv2.VideoCapture(available_cameras[0])
                            # Open with default settings first
                            for _ in range(5):  # Try reading a few frames to initialize
                                instance.cap.read()
                                time.sleep(0.05)
                            # Then configure
                            instance.cap = configure_camera(instance.cap)
                            # Reset error counters
                            error_count = 0
                            consecutive_errors = 0
                        else:
                            # If no camera available, use virtual camera
                            instance.cap = create_virtual_camera()
                            
                        # Read a few frames to ensure camera is working
                        for _ in range(5):
                            instance.cap.read()
                            time.sleep(0.05)
                            
                    except Exception as e:
                        print(f"Error resetting camera: {e}")
                        instance.cap = create_virtual_camera()
                
                # If frame read failed, generate a blank frame
                if not ret or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera Error - Reconnecting...", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Reset error count if frame was successfully read
                error_count = 0
                consecutive_errors = 0
                
                # Store successful frame
                last_successful_frame = frame.copy()
                
                # Apply frame interpolation if enabled
                if INTERPOLATE_FRAMES and prev_frame is not None:
                    try:
                        # Calculate optical flow for interpolation
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate dense optical flow
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        
                        # Generate interpolated frames
                        for i in range(1, MAX_INTERPOLATION_FRAMES + 1):
                            alpha = i / (MAX_INTERPOLATION_FRAMES + 1)
                            
                            # Use a more efficient method for interpolation
                            h, w = frame.shape[:2]
                            flow_map = np.zeros((h, w, 2), np.float32)
                            flow_map[:,:,0] = alpha * flow[:,:,0]
                            flow_map[:,:,1] = alpha * flow[:,:,1]
                            
                            # Warp the previous frame using the flow
                            interpolated = cv2.remap(prev_frame, 
                                                  (np.identity(3) + flow_map[:,:,::-1]).astype(np.float32),
                                                  None, cv2.INTER_LINEAR)
                            
                            interpolation_buffer.append(interpolated)
                    except Exception as e:
                        print(f"Error in frame interpolation: {e}")
                
                # Process the frame
                try:
                    # Apply basic image enhancement
                    frame = enhance_image(frame)
                    
                    # Resize frame if needed
                    if RESIZE_FACTOR != 1.0:
                        new_width = int(frame.shape[1] * RESIZE_FACTOR)
                        new_height = int(frame.shape[0] * RESIZE_FACTOR)
                        frame = cv2.resize(frame, (new_width, new_height), 
                                          interpolation=cv2.INTER_LINEAR)
                    
                    # Process frame for object detection if enabled
                    if args.process_frames and process_this_frame:
                        try:
                            processed_frame, detections, hazards = instance.process_frame(frame)
                            
                            # Store results for use with skipped frames
                            last_detections = detections
                            last_hazards = hazards
                            if instance.safe_path:
                                last_safe_path = instance.safe_path
                                
                        except Exception as e:
                            print(f"Error processing frame: {e}")
                            processed_frame = frame.copy()
                            cv2.putText(processed_frame, "Processing Error", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # For skipped frames, use the original frame but draw the last detections
                        processed_frame = frame.copy()
                        
                        if not args.process_frames:
                            cv2.putText(processed_frame, "Processing Disabled", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        elif not process_this_frame:
                            cv2.putText(processed_frame, "Lightweight Mode", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
                            
                            # Draw last known detections on this frame
                            for obj in last_detections:
                                if 'box' in obj:
                                    x1, y1, x2, y2 = obj['box']
                                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Display object info with distance
                                    label = f"{obj['class']}"
                                    if obj.get('distance'):
                                        label += f" {obj['distance']:.1f}m"
                                    label += f" {obj['direction']}"
                                    
                                    cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Store current frame for next iteration
                    prev_frame = frame.copy()
                    
                    # Prepare frame for web delivery
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not ret:
                        raise Exception("Failed to encode frame as JPEG")
                    
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in the format required by Flask
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Yield interpolated frames if available
                    while interpolation_buffer:
                        interp_frame = interpolation_buffer.pop(0)
                        ret, buffer = cv2.imencode('.jpg', interp_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    # Create a simple error frame
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, f"Video Error: {str(e)[:30]}", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error in main frame generation loop: {e}")
            time.sleep(0.1)  # Shorter sleep on error
            continue

def enhance_image(frame):
    """Apply image enhancement to improve quality and detection accuracy"""
    try:
        # Skip if frame is None or empty
        if frame is None or frame.size == 0:
            return frame
        
        # Create a copy to avoid modifying the original
        enhanced = frame.copy()
        
        if COLOR_CORRECTION:
            # Convert to HSV for better color manipulation
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            
            # Boost saturation for more vivid colors
            hsv[:,:,1] = np.clip(hsv[:,:,1] * SATURATION_BOOST, 0, 255).astype(np.uint8)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Apply contrast and brightness adjustment for better visibility
            enhanced = cv2.convertScaleAbs(enhanced, alpha=CONTRAST_BOOST, beta=BRIGHTNESS_BOOST)
        
            # Apply mild sharpening if resolution is high enough
            if frame.shape[1] > 640:
                kernel = np.array([[-0.5, -0.5, -0.5],
                                [-0.5, 5, -0.5],
                                [-0.5, -0.5, -0.5]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                
            # Apply auto white balance
            wb = cv2.xphoto.createSimpleWB() if hasattr(cv2, 'xphoto') else None
            if wb is not None:
                enhanced = wb.balanceWhite(enhanced)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return frame

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/scene_description', methods=['GET'])
def scene_description():
    """API endpoint to get scene description"""
    instance = get_nayan_instance()
    description = instance.cmd_describe_scene()
    return jsonify({'description': description})

@app.route('/api/identify_objects', methods=['GET'])
def identify_objects():
    """API endpoint to identify objects"""
    instance = get_nayan_instance()
    objects = instance.cmd_identify_objects()
    return jsonify({'objects': objects})

@app.route('/api/read_text', methods=['GET'])
def read_text():
    """API endpoint to activate OCR"""
    instance = get_nayan_instance()
    result = instance.cmd_read_text()
    return jsonify({'status': result})

@app.route('/api/navigate', methods=['GET'])
def navigate():
    """API endpoint for navigation"""
    instance = get_nayan_instance()
    guidance = instance.cmd_navigate()
    return jsonify({'guidance': guidance})

@app.route('/api/commands', methods=['GET'])
def help_commands():
    """API endpoint to list available commands"""
    instance = get_nayan_instance()
    help_text = instance.cmd_help()
    return jsonify({'commands': help_text})

@app.route('/api/toggle_mode', methods=['GET'])
def toggle_mode():
    """Toggle between lightweight and full processing modes"""
    global LIGHTWEIGHT_MODE
    LIGHTWEIGHT_MODE = not LIGHTWEIGHT_MODE
    mode = "lightweight" if LIGHTWEIGHT_MODE else "full processing"
    socketio.emit('announcement', {'text': f"Switched to {mode} mode", 'priority': True})
    return jsonify({'mode': mode})

@app.route('/sounds/<filename>')
def serve_sound(filename):
    """Serve sound files"""
    # Get the Nayan instance to access the sound directory
    instance = get_nayan_instance()
    # Send the file from the sounds directory
    sound_dir = instance.audio_manager.sound_dir
    return Response(open(os.path.join(sound_dir, filename), 'rb').read(), 
                   mimetype='audio/mpeg')

@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    """Check camera status"""
    instance = get_nayan_instance()
    
    # Check if camera is opened
    is_opened = instance.cap.isOpened()
    
    # Try to grab a frame
    ret = False
    frame_size = None
    if is_opened:
        ret, frame = instance.cap.read()
        if ret and frame is not None:
            frame_size = frame.shape
            # Release the frame back
            frame = None
    
    # Get camera properties
    if is_opened:
        cam_width = instance.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cam_height = instance.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cam_fps = instance.cap.get(cv2.CAP_PROP_FPS)
    else:
        cam_width = 0
        cam_height = 0 
        cam_fps = 0
    
    return jsonify({
        'camera_opened': is_opened,
        'frame_read': ret,
        'frame_size': frame_size,
        'camera_properties': {
            'width': cam_width,
            'height': cam_height,
            'fps': cam_fps
        },
        'settings': {
            'camera_index': args.camera_index,
            'using_sample': args.use_sample_video,
            'resize_factor': RESIZE_FACTOR,
            'fps_limit': FPS,
            'process_frames': args.process_frames,
            'lightweight_mode': LIGHTWEIGHT_MODE,
            'process_every_n_frames': PROCESS_EVERY_N_FRAMES
        }
    })

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print('Client connected')
    # Send test notification
    socketio.emit('announcement', {'text': 'Connected to Nayan server', 'priority': True})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nayan Application')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--camera-index', dest='camera_index', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--use-sample-video', dest='use_sample_video', action='store_true', help='Use sample video instead of camera')
    parser.add_argument('--fps', type=int, default=24, help='Target frames per second (default: 24)')
    parser.add_argument('--resize', type=float, default=0.75, help='Resize factor for video (default: 0.75, range 0.1-1.0)')
    parser.add_argument('--camera-width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--camera-height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--process-frames', dest='process_frames', action='store_true', help='Enable full frame processing')
    parser.add_argument('--no-process-frames', dest='process_frames', action='store_false', help='Disable full frame processing')
    parser.add_argument('--lightweight', dest='lightweight', action='store_true', help='Use lightweight mode (default: false)')
    parser.add_argument('--process-every', type=int, default=5, help='Process every N frames in lightweight mode (default: 5)')
    parser.add_argument('--jpeg-quality', type=int, default=85, help='JPEG quality for streaming (0-100, default: 85)')
    parser.set_defaults(process_frames=True, lightweight=True)
    
    args = parser.parse_args()
    
    print(f"Starting Nayan with camera index {args.camera_index}, FPS={args.fps}, resize={args.resize}")
    print(f"Visit http://{args.host}:{args.port}/ in your browser to access the interface")
    
    # Set FPS and resize factor without using global statement
    # (These are actually global module variables)
    FPS = args.fps
    RESIZE_FACTOR = max(0.1, min(1.0, args.resize))  # Limit between 0.1 and 1.0
    LIGHTWEIGHT_MODE = args.lightweight
    PROCESS_EVERY_N_FRAMES = max(1, args.process_every)
    JPEG_QUALITY = max(50, min(100, args.jpeg_quality))  # Limit between 50-100
    
    print(f"Lightweight mode: {'Enabled' if LIGHTWEIGHT_MODE else 'Disabled'}")
    if LIGHTWEIGHT_MODE:
        print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames")
    
    # Detect available cameras
    available_cameras = detect_available_cameras()
    
    # Initialize Nayan instance
    nayan = get_nayan_instance()
    
    # Run Flask app with SocketIO
    socketio.run(app, host=args.host, port=args.port, debug=args.debug) 