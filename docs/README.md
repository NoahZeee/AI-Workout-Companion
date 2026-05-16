# Real-Time Workout Form Analyzer

Pose-based rep counter and form validator using MediaPipe. Supports push-ups and bicep curls from side-view footage.

## Quick Start

```bash
# Webcam analysis
python run.py --exercise push-ups

# Analyze video file
python run.py --exercise bicep-curls --input workout.mp4 --output outputs/analysis.mp4 --no-display

# Get help
python run.py --help
```

## Requirements

**System Requirements:**
- Python 3.8 or higher
- 2GB RAM minimum
- Webcam or video file input
- Windows, macOS, or Linux

**Dependencies:**
- `opencv-python` (≥4.5.0) - Video I/O and frame processing
- `mediapipe` (≥0.10.0) - Pose detection
- `numpy` (≥1.19.0) - Numerical operations

## Installation

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python mediapipe numpy
```

MediaPipe automatically downloads its pose model (~5MB) on first run.

## Usage

### Options
```
--exercise {push-ups, bicep-curls}  Exercise type (default: push-ups)
--input INPUT                        Video source: "0" (webcam), device ID, or file path (default: 0)
--output OUTPUT                      Output video path (default: outputs/timestamp.mp4)
--no-display                         Skip live video display (faster)
--confidence FLOAT                   Pose detection confidence 0-1 (default: 0.7)
```

### Examples

```bash
# Webcam with display
python run.py --exercise push-ups

# Process video, no display
python run.py --exercise push-ups --input video.mp4 --output outputs/result.mp4 --no-display

# Bicep curls from webcam, save output
python run.py --exercise bicep-curls --output outputs/curls.mp4

# Higher detection confidence
python run.py --confidence 0.8 --input workout.mp4
```

## How It Works

### Pipeline
```
Video Input → Pose Detection (MediaPipe) → Exercise Analysis 
→ Rep Counting & Form Validation → Feedback Annotation → Output
```

### Rep Detection
Uses **state machine** based on joint angles:
- **Push-ups**: Tracks elbow angle (extended/flexed states)
- **Bicep Curls**: Tracks elbow flexion and alignment with shoulder

### Form Validation
- **Push-ups**: Validates depth (elbow ≤90°) and hip alignment (body line)
- **Bicep Curls**: Validates flex depth (≤80°) and elbow stays below shoulder

## Architecture

### Core Modules (`src/`)

**main.py**
- CLI entry point with argument parsing
- Orchestrates component initialization

**pose_detector.py**
- MediaPipe Pose Landmarker integration
- Converts 33-point format → 17-point COCO
- Provides angle calculations and pose drawing

**exercise_analyzer.py**
- Base `ExerciseAnalyzer` class
- `PushUpAnalyzer`: Push-up rep counting and form validation
- `BicepCurlAnalyzer`: Bicep curl analysis
- Extensible for new exercises

**feedback_generator.py**
- Real-time frame annotation
- Rep counter display
- Form feedback messages
- Performance metrics visualization

**video_processor.py**
- Main processing loop
- Handles webcam and file I/O
- Coordinates detection → analysis → annotation

### Project Structure
```
├── run.py                    # Entry point
├── requirements.txt          # Python dependencies
├── src/
│   ├── main.py              # CLI
│   ├── pose_detector.py     # Pose detection
│   ├── exercise_analyzer.py # Rep counting
│   ├── feedback_generator.py # Visualization
│   └── video_processor.py   # Video pipeline
├── docs/
│   └── README.md            # This file
├── data/                     # Video files for analysis
└── outputs/                 # Generated analysis videos
```

## Customization

### Tune Detection Thresholds

Edit `src/exercise_analyzer.py`:

```python
# Push-ups
ELBOW_DOWN_THRESHOLD = 100      # Arm flexed
ELBOW_UP_THRESHOLD = 150        # Arm extended
MIN_DEPTH_ANGLE = 90            # Minimum elbow bend for valid rep
HIP_SAG_TOLERANCE = 0.40        # Max body line deviation

# Bicep Curls
ELBOW_DOWN_THRESHOLD = 140      # Arm extended
ELBOW_UP_THRESHOLD = 80         # Arm flexed
MIN_FLEX_ANGLE = 80             # Minimum elbow bend for valid rep
ELBOW_ALIGNMENT_TOLERANCE = 0.60  # Max deviation from vertical
```

### Add a New Exercise

1. Create analyzer in `src/exercise_analyzer.py`:

```python
class SquatAnalyzer(ExerciseAnalyzer):
    def analyze(self, landmarks, visibility):
        # Calculate knee angle, hip angle, etc.
        # Implement rep counting state machine
        # Validate form
        return {'rep_count': count, 'feedback': messages, ...}
    
    def get_required_landmarks(self):
        return ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE', ...]
```

2. Register in `src/main.py`:

```python
elif args.exercise == 'squats':
    analyzer = SquatAnalyzer(pose_detector)
```

3. Run:
```bash
python run.py --exercise squats --input video.mp4
```

## Performance

| Metric | Value |
|--------|-------|
| Pose Detection | ~33-36 FPS |
| Model Size | ~5 MB |
| Keypoints | 33 (MediaPipe) → 17 (COCO) |
| Confidence Threshold | 0.7 (adjustable) |

## Troubleshooting

**No reps counted?**
- Check angle thresholds in `src/exercise_analyzer.py`
- Try `--confidence 0.6` for looser detection
- Ensure camera is side-view

**Low FPS?**
- Normal - real pose detection is computationally intensive
- Use `--no-display` for faster processing
- Try lower confidence threshold

**Video won't open?**
- Use absolute path to video file
- Verify file exists and is readable
- Try different codec (MP4/H.264)

**ModuleNotFoundError?**
- Run `pip install opencv-python mediapipe numpy`
- Use Python 3.8+

## Limitations and Special Instructions

### Camera Requirements
- **Optimal angle**: Side-view camera (90° to body plane)
- **Arm visibility**: Entire arm must be visible for analysis
- **Single subject**: Designed for one person per video
- **Lighting**: Good lighting recommended for pose detection accuracy

### Detection Constraints
- Minimum 50% pose detection confidence required for frame inclusion
- Works best with clear, unobstructed body position
- Does not detect multi-person workouts
- No support for exercises with both arms performing the same motion (e.g., pushups vs diamond pushups)

### Output Specifications
- **Video Codec**: H.264 (MP4 format)
- **Frame Rate**: Maintains input frame rate
- **Resolution**: Supports any resolution; adaptive UI scaling for portrait/landscape
- **File Size**: Output file size depends on input length and resolution

### Processing Notes
- **Real-time performance**: 33-36 FPS typical on modern hardware
- **First run**: ~30-60 seconds for MediaPipe model download and initialization
- **No GPU required**: Works on CPU; optional GPU acceleration available through MediaPipe
- **Stateless**: Each frame analyzed independently; no multi-frame buffering

### Form Validation Accuracy
- Thresholds calibrated for natural, controlled movement
- May require fine-tuning for individual body types or movement styles
- Form feedback validates primary plane of motion (side-view optimized)
- Provide clear, consistent movement patterns for best rep counting accuracy

## References

- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [OpenCV](https://docs.opencv.org/)
- [NumPy](https://numpy.org/doc/)
