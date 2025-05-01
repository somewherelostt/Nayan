# Nayan - AI-Powered Vision Assistant

Nayan is an advanced AI-powered vision assistant that provides real-time object detection, scene description, and navigation assistance for visually impaired individuals.

## Features

- Real-time object detection and recognition with stable object tracking
- Scene description and analysis
- Text reading (OCR) capabilities
- Navigation assistance and hazard detection
- High-quality video streaming with frame interpolation
- Voice feedback and announcements
- Web-based interface with responsive design

## Key Improvements

- **Enhanced Video Quality**: Optimized video pipeline with motion interpolation
- **Stable Object Detection**: Implemented object tracking to reduce flickering and improve consistency
- **Performance Optimizations**:
  - Hardware acceleration
  - Frame rate control
  - Optimized image processing
  - Memory management
- **Improved Object Recognition**:
  - Better confidence thresholds
  - Additional reference objects
  - Enhanced distance estimation

## Requirements

- Python 3.8 or higher
- OpenCV 4.5 or higher
- Flask 2.0 or higher
- Flask-SocketIO
- NumPy
- A compatible webcam or camera device

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/somewherelostt/NayanV0.git
   cd nayan
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Run the application:

   ```
   python main.py
   ```

4. Open your web browser and navigate to:

   ```
   http://localhost:5000
   ```

## Configuration

The application can be configured using command-line arguments:

```bash
python main.py --camera-index 0 --fps 30 --resize 0.75 --process-frames
```

Available options:

- `--camera-index`: Specify camera device index (default: 0)
- `--fps`: Set target frames per second (default: 30)
- `--resize`: Set video resize factor (default: 0.75)
- `--process-frames`: Enable full frame processing
- `--lightweight`: Use lightweight mode for better performance
- `--process-every`: Process every N frames in lightweight mode
- `--jpeg-quality`: Set JPEG quality for streaming (0-100)

## Performance Optimization

The application includes several optimizations for better performance:

- Hardware acceleration support
- Frame rate control
- Resolution optimization
- Image enhancement
- Noise reduction
- Motion JPEG codec for better quality
- Frame interpolation for smoother video
- Object tracking for more stable detection

## Troubleshooting

If you experience issues:

1. Check camera compatibility
2. Verify all dependencies are installed
3. Try different camera settings
4. Check system resources
5. Review the debug logs
6. Try reducing resolution or FPS if performance is poor

## Acknowledgments

- OpenCV for computer vision capabilities
- Flask for web framework
