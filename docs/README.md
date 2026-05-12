# Real-Time Workout Form Analyzer

A Python application optimized for **side-view workout footage** that uses pose estimation to provide real-time feedback, count reps, and analyze exercise form. Built with **MediaPipe Pose Landmarker** and OpenCV.

## Features

- **Real-Time Pose Detection**: Uses MediaPipe Pose Landmarker for accurate body pose estimation
- **Side-View Optimized**: Works with single-side body visibility (left arm OR right arm visible)
- **Rep Counting**: Automatically counts reps based on exercise-specific angle thresholds
- **Form Feedback**: Provides real-time guidance on proper form (depth, back alignment)
- **Live Video Display**: Shows pose skeleton (17 keypoints), rep counter, and feedback overlaid on video
- **Output Video**: Saves annotated video with all feedback and analysis for later review
- **Extensible Architecture**: Easy to add new exercises by extending the `ExerciseAnalyzer` class
- **Automatic Model Download**: MediaPipe model downloads automatically on first run (~5 MB)

## Supported Exercises

- **Push-ups**: Tracks elbow flexion, back alignment, and depth (side-view optimized)
- **Bicep Curls**: Tracks elbow angle from side perspective
- **Extensible**: Add squats, deadlifts, pull-ups, planks, etc. (framework ready)

## Installation

### Requirements
- Python 3.8+
- pip

### Quick Setup

```bash
# Install dependencies (one command)
pip install opencv-python mediapipe numpy
```

That's it! MediaPipe model downloads automatically on first run.

## Quick Start

### Webcam Analysis (Recommended)

```bash
# Analyze push-ups with your webcam
python run.py --exercise push-ups
```

See your real pose detected in real-time!

### Analyze Your Video

```bash
# Process your workout video
python run.py --input your_video.mp4 --output outputs/analysis.mp4
```

### Test with Sample Video

```bash
# Process included test video
python run.py --input data/test_video.mp4 --output outputs/test_analysis.mp4 --no-display
```

## Usage Guide

### Entry Point

Use `run.py` from the project root (automatically sets up Python path):

```bash
python run.py [OPTIONS]
```

Alternatively, from the `src/` directory:

```bash
cd src
python main.py [OPTIONS]
```

### All Options

```bash
python run.py --help
```

Options:
- `--exercise {push-ups, bicep-curls}`: Exercise type (default: push-ups)
- `--input INPUT`: Video source - "0" for webcam, device ID, or file path (default: "0" - webcam)
- `--output OUTPUT`: Path to save output video (saved to `outputs/` folder by default if no path given)
- `--no-display`: Don't show live video window (faster processing)
- `--confidence CONFIDENCE`: Minimum pose detection confidence 0-1 (default: 0.7)

## Examples

```bash
# Webcam push-up analysis with display
python run.py --exercise push-ups

# Analyze video file, save to outputs/
python run.py --exercise push-ups --input my_workout.mp4 --output outputs/analysis.mp4

# Bicep curls from webcam, save without display
python run.py --exercise bicep-curls --output outputs/biceps.mp4 --no-display

# High-confidence detection (more accurate, slower)
python run.py --confidence 0.8 --input workout.mp4

# Process multiple videos (no display = faster)
python run.py --input video1.mp4 --output outputs/v1.mp4 --no-display
python run.py --input video2.mp4 --output outputs/v2.mp4 --no-display
```

## How It Works

### Architecture

1. **PoseDetector** (`pose_detector.py`)
   - Wraps MediaPipe Pose for pose estimation
   - Extracts 33 3D body landmarks from each frame
   - Computes angles between joints (e.g., elbow angle)
   - Draws pose skeleton on frames

2. **ExerciseAnalyzer** (`exercise_analyzer.py`)
   - Base class for exercise-specific logic
   - `PushUpAnalyzer`: Detects push-ups, validates form
   - `BicepCurlAnalyzer`: Detects bicep curls
   - Extensible design for new exercises

3. **FeedbackGenerator** (`feedback_generator.py`)
   - Formats and displays real-time feedback text
   - Annotates frames with rep counter, form warnings
   - Generates summary frames

4. **VideoProcessor** (`video_processor.py`)
   - Main processing loop
   - Handles webcam and file input/output
   - Coordinates pose detection → analysis → feedback

5. **main.py**
   - CLI entry point
   - Argument parsing and setup

## Project Structure

```
Project Root/
├── run.py                      # Entry point (use this!)
├── src/                        # Source code
│   ├── main.py                # CLI interface
│   ├── pose_detector.py       # MediaPipe integration
│   ├── exercise_analyzer.py   # Rep counting & form validation
│   ├── feedback_generator.py  # Video annotation
│   └── video_processor.py     # Video I/O pipeline
├── docs/                       # Documentation
│   ├── README.md              # This file
│   ├── QUICKSTART.md          # Quick start guide
│   ├── PROJECT_SUMMARY.md     # Architecture overview
│   └── REAL_DETECTION_READY.md # MediaPipe details
├── data/                       # Test data
│   ├── test_video.mp4         # Sample video (150 frames)
│   ├── create_test_video.py   # Video generator
│   └── pose_landmarker_lite.task # MediaPipe model (auto-downloaded)
└── outputs/                    # Generated analysis videos
```

For detailed file descriptions, see [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md).

## How It Works

### Real-Time Pipeline

```
Video Input (webcam or file)
    ↓
Frame Stream (30 FPS)
    ↓
MediaPipe Pose Detection (33 keypoints)
    ↓
Convert to 17-point COCO format
    ↓
Exercise Analysis (angle-based state machine)
    ↓
Rep Counting & Form Validation
    ↓
Feedback Generation
    ↓
Video Annotation & Output
```

### Rep Detection (Push-ups Example)

Reps are detected using a **state machine** based on elbow angle:

```
State: UP (elbow angle > 150°)
  ↓
  Detect elbow angle < 100° → Switch to DOWN
State: DOWN (in push-up position)
  ↓
  Detect elbow angle > 150° → Switch to UP, COUNT REP
  ↓
Rep counter incremented
```

### Form Validation (Push-ups)

Real-time form checks include:
- **Depth**: Elbow must bend adequately for full rep
- **Back Alignment**: Spine should remain neutral (not sagging or arching)

## Performance

| Metric | Value |
|--------|-------|
| Pose Detection | ~73 FPS on CPU |
| Model Size | ~5 MB |
| Processing Time (150 frames) | ~2 seconds |
| Confidence Threshold | 0.7 (adjustable) |
| Keypoints | 33-point MediaPipe → 17-point COCO |

## Customization

### Tune Angle Thresholds

Edit `src/exercise_analyzer.py`:

```python
class PushUpAnalyzer(ExerciseAnalyzer):
    ELBOW_DOWN_THRESHOLD = 100   # Adjust for your needs
    ELBOW_UP_THRESHOLD = 150     # Lower = easier reps
```

### Add a New Exercise

1. Create analyzer in `src/exercise_analyzer.py`:

```python
class SquatAnalyzer(ExerciseAnalyzer):
    def analyze(self, landmarks, visibility):
        # Calculate knee and hip angles
        # Implement state machine
        # Return rep count and form feedback
        pass
```

2. Register in `src/main.py`:

```python
elif args.exercise == 'squats':
    analyzer = SquatAnalyzer(pose_detector)
```

3. Run:

```bash
python run.py --exercise squats
```

## Troubleshooting

**Q: "ModuleNotFoundError: mediapipe"**  
A: Install it: `pip install mediapipe`

**Q: "No module named 'cv2'"**  
A: Install OpenCV: `pip install opencv-python`

**Q: "Cannot open video file"**  
A: Check path is correct and file exists. Use absolute paths if relative fails.

**Q: "Low FPS (< 20)"**  
A: Normal! Real MediaPipe detection uses more compute. Use `--no-display` for faster processing.

**Q: "0 reps counted"**  
A: Check angle thresholds match your form. Try lowering them in `exercise_analyzer.py`.

**Q: "Download takes forever"**  
A: Normal on first run (~2 min for ~5 MB model). Subsequent runs use cached model.

**Q: "Pose detection unstable"**  
A: Try higher confidence: `--confidence 0.8` (more stable, fewer detections)

## Next Steps

1. **Test with webcam**: `python run.py --exercise push-ups`
2. **Analyze your videos**: Use your own workout footage
3. **Fine-tune thresholds**: Adjust angles in `src/exercise_analyzer.py`
4. **Add exercises**: Implement squats, deadlifts, etc.
5. **Integrate**: Use pose_detector.py as a library

## Support

For detailed information, see:
- [QUICKSTART.md](QUICKSTART.md) - Command examples and quick setup
- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - File organization


## References

- [MediaPipe Pose Documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
