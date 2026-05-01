# Quick Start Guide

## 🚀 Run with Real MediaPipe Pose Detection

```bash
# Test with real pose detection (model auto-downloads on first run)
python run.py --input data/test_video.mp4 --output outputs/my_analysis.mp4 --no-display
```

**What it does:**
- Uses real MediaPipe Pose Landmarker for accurate pose detection
- Analyzes push-ups (default exercise)
- Counts reps using real body joint angles
- Saves annotated video with skeleton overlay and feedback
- Performance: ~73 FPS with real detection

## 📹 Analyze Your Workout Video

```bash
# Analyze push-ups from your video
python run.py --exercise push-ups --input my_workout.mp4 --output outputs/analysis.mp4

# Analyze bicep curls
python run.py --exercise bicep-curls --input video.mp4 --output outputs/output.mp4

# No display (faster processing)
python run.py --input video.mp4 --output outputs/output.mp4 --no-display
```

## 🎥 Live Webcam Analysis (Recommended!)

See real poses detected on your own body:

```bash
# Start webcam analysis for push-ups
python run.py --exercise push-ups

# Save webcam session to file
python run.py --exercise push-ups --output outputs/my_session.mp4

# Bicep curls with webcam
python run.py --exercise bicep-curls
```

## 📊 All Commands

```bash
# Get help
python run.py --help

# Specific options
python run.py --exercise push-ups --input 0 --output outputs/output.mp4 --confidence 0.8

# Alternative: run from src/ directly
cd src
python main.py --exercise push-ups
```

## 📁 Project Structure

```
├── run.py                  ← START HERE (entry point)
├── src/                    ← Source code
│   ├── main.py
│   ├── pose_detector.py
│   ├── exercise_analyzer.py
│   ├── feedback_generator.py
│   └── video_processor.py
├── docs/                   ← Documentation
│   ├── README.md           ← Full user guide
│   ├── QUICKSTART.md
│   ├── REAL_DETECTION_READY.md
│   └── PROJECT_SUMMARY.md
├── data/                   ← Test data
│   ├── test_video.mp4
│   ├── pose_landmarker_lite.task  (auto-downloaded)
│   └── create_test_video.py
└── outputs/                ← Generated videos
```

**See PROJECT_STRUCTURE.md for detailed file descriptions**

## 🔄 Next: Extend the System

**Real pose detection is already integrated!** MediaPipe Pose Landmarker is running automatically.

### Add More Exercises

Extend `src/exercise_analyzer.py`:

```python
class SquatAnalyzer(ExerciseAnalyzer):
    """Analyze squat form using knee and hip angles"""
    
    def analyze(self, landmarks, visibility):
        # Get key points
        left_knee_angle = self.pose_detector.calculate_angle(
            landmarks[11], landmarks[13], landmarks[15]  # hip, knee, ankle
        )
        # Count reps based on angle thresholds
        # Validate form
```

Then add to CLI in `src/main.py`:
```python
choices=['push-ups', 'bicep-curls', 'squats']
```

### Tune Angle Thresholds

Edit `src/exercise_analyzer.py`:
```python
ELBOW_DOWN_THRESHOLD = 100   # Adjust for your body
ELBOW_UP_THRESHOLD = 150     # Adjust for your range
```

## ✅ What's Ready Now

- [x] Full pipeline (real pose → analysis → feedback → video output)
- [x] Real MediaPipe Pose Landmarker integration
- [x] Automatic model download (5 MB)
- [x] Rep counting with angle-based state machine
- [x] Form feedback system (depth, alignment, symmetry)
- [x] Real-time visualization (73 FPS)
- [x] Video file and webcam input/output
- [x] Clean CLI interface
- [x] Extensible exercise system
- [x] Organized project structure

## 🎯 You're Ready to Go!

Run `python run.py --exercise push-ups` and start analyzing!
