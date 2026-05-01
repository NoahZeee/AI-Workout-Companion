# Project Structure & Organization

## Folder Layout

```
424 Final/
├── src/                          # Source code
│   ├── main.py                  # Entry point (main CLI interface)
│   ├── pose_detector.py         # MediaPipe pose detection integration
│   ├── exercise_analyzer.py     # Rep counting & form validation
│   ├── feedback_generator.py    # Real-time feedback visualization
│   └── video_processor.py       # Video I/O and processing pipeline
│
├── docs/                        # Documentation
│   ├── README.md               # Full user guide and features
│   ├── QUICKSTART.md           # Quick start examples
│   ├── REAL_DETECTION_READY.md # Real pose detection setup
│   └── PROJECT_SUMMARY.md      # Architecture & design overview
│
├── data/                        # Test data and models
│   ├── test_video.mp4          # Sample test video (150 frames)
│   ├── pose_landmarker_lite.task  # MediaPipe model (auto-downloaded)
│   └── create_test_video.py    # Utility to generate synthetic test videos
│
├── outputs/                     # Generated workout analysis videos
│   └── (user-generated files)
│
├── run.py                       # Convenient entry point (runs src/main.py)
└── .vscode/                     # VS Code settings (kept from original)
```

## Quick Start

```bash
# From project root, any of these work:
python run.py --exercise push-ups
python run.py --input data/test_video.mp4 --output outputs/analysis.mp4 --no-display
```

## File Descriptions

### Source Code (`src/`)

**main.py** (161 lines)
- CLI entry point with argparse configuration
- Handles: --exercise, --input, --output, --no-display, --confidence flags
- Orchestrates initialization of all components
- Calls: PoseDetector → ExerciseAnalyzer → VideoProcessor

**pose_detector.py** (250+ lines)
- Real MediaPipe Pose Landmarker integration
- Auto-downloads model on first run (data/pose_landmarker_lite.task)
- Converts 33-point MediaPipe format → 17-point COCO format
- Graceful fallback: synthetic poses if MediaPipe unavailable
- Key methods:
  - `detect_pose(frame)` - returns (landmarks, visibility)
  - `calculate_angle()` - computes joint angles
  - `draw_pose()` - draws skeleton overlay

**exercise_analyzer.py** (300+ lines)
- Abstract analyzer base class
- PushUpAnalyzer: angle-based rep counting, form validation
- BicepCurlAnalyzer: arm angle tracking
- Extensible: add SquatAnalyzer, DeadliftAnalyzer, PullUpAnalyzer easily
- Key methods:
  - `analyze()` - processes pose and counts reps
  - `get_required_landmarks()` - specifies which joints needed
  - Form validation: depth, alignment, symmetry

**feedback_generator.py** (250+ lines)
- Real-time overlay generation
- Rep counter display
- Form warning messages
- Angle metrics visualization
- Session summary frame generation
- Key method: `annotate_frame()` - main annotation orchestrator

**video_processor.py** (300+ lines)
- Video input/output abstraction
- Supports: webcam (device ID), MP4 files
- Main processing loop: frame capture → pose detection → analysis → annotation → output
- MP4v codec for video writing
- Key method: `process_video()` - main pipeline

### Documentation (`docs/`)

**README.md** - Full user guide, installation, features, troubleshooting
**QUICKSTART.md** - Command examples and quick reference
**REAL_DETECTION_READY.md** - Real MediaPipe detection setup and usage
**PROJECT_SUMMARY.md** - Architecture overview and design decisions

### Data (`data/`)

**test_video.mp4** - Sample 150-frame test video (5 seconds @ 30 FPS, 640x480)
**pose_landmarker_lite.task** - MediaPipe Pose Landmarker model (~5 MB, auto-downloaded)
**create_test_video.py** - Generate synthetic test videos with opencv-python

## Running the Program

### From Root Directory
```bash
python run.py [options]
```

### From Source Directory (less convenient)
```bash
cd src
python main.py [options]
```

### Commands
```bash
# Webcam with push-ups (display enabled)
python run.py --exercise push-ups

# Analyze video file
python run.py --input data/test_video.mp4 --output outputs/my_analysis.mp4 --no-display

# Webcam with bicep curls, save output
python run.py --exercise bicep-curls --output outputs/bicep_session.mp4

# Help
python run.py --help
```

## What Was Cleaned Up

### Deleted Obsolete Files
- `test_mediapipe.py` - Old MediaPipe testing (real integration now in place)
- `download_model.py` - No longer needed (auto-download built in)
- `check_mediapipe.py` - Verification not needed
- `REAL_POSE_SETUP.py` - Reference file (implementation done)
- `IMPLEMENTATION_GUIDE.md` - Outdated (MediaPipe integrated)
- `openpose_pose.prototxt.txt` - Old OpenPose config (unused)
- `video.mp4` - Old test file
- All test output videos (*.mp4 from root, regenerable)
- `__pycache__/` - Auto-generated

### Organized into Folders
- Core modules → `src/`
- Documentation → `docs/`
- Test data & models → `data/`
- Generated outputs → `outputs/`

## Performance

Real MediaPipe detection: **~73 FPS** (real computation)
- Model inference: ~13ms per frame
- Compatible with live webcam feeds and video files

## Next Steps

1. **Test with your own videos** - See real pose detection in action
2. **Fine-tune thresholds** - In `src/exercise_analyzer.py`:
   - Adjust ELBOW_DOWN_THRESHOLD and ELBOW_UP_THRESHOLD
   - Tweak form validation sensitivity
3. **Add more exercises** - Extend `ExerciseAnalyzer` base class:
   - Squats (knee & hip angles)
   - Deadlifts (back angle)
   - Curls (full arm range)
   - Planks (back alignment)

## Architecture Summary

```
Video Input (webcam or file)
    ↓
Frame Processing Loop (30 FPS)
    ↓
MediaPipe Pose Detection (33 keypoints)
    ↓
Map to COCO 17-point format (compatibility)
    ↓
Exercise Analysis (angle state machine)
    ↓
Rep Counting & Form Validation
    ↓
Real-time Feedback Generation
    ↓
Video Annotation & Output
```

---
**Status**: Production Ready
**Real Detection**: Active (MediaPipe Pose Landmarker)
**Ready to Use**: Yes
