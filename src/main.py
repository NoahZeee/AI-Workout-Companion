#!/usr/bin/env python3
"""
Real-Time Workout Form Analyzer
Analyzes exercise form using pose estimation, counts reps, and provides real-time feedback.

Usage:
    python main.py --exercise push-ups                          # Webcam + push-ups
    python main.py --exercise push-ups --input video.mp4        # Video file + push-ups
    python main.py --exercise bicep-curls --input 0 --output output.mp4
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from pose_detector import PoseDetector
from exercise_analyzer import PushUpAnalyzer, BicepCurlAnalyzer
from video_processor import VideoProcessor


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Workout Form Analyzer with Rep Counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --exercise push-ups
    Uses webcam (device 0) for push-up analysis, displays live video

  python main.py --exercise push-ups --input video.mp4 --output output.mp4
    Analyzes push-ups in video.mp4, saves annotated output to output.mp4

  python main.py --exercise bicep-curls --input 0
    Uses webcam for bicep curl analysis
        """
    )
    
    parser.add_argument(
        '--exercise',
        type=str,
        choices=['push-ups', 'bicep-curls'],
        default='push-ups',
        help='Exercise to analyze (default: push-ups)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='0',
        help='Video input source: device ID (default 0) or file path (default: webcam)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video file path. If not specified, video is not saved.'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display live video window'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Minimum pose detection confidence (0-1, default: 0.7)'
    )
    
    return parser.parse_args()


def create_output_path(base_name: str = "workout_output") -> str:
    """
    Create output file path with timestamp if not specified.
    Saves to outputs/ folder by default.
    
    Args:
        base_name: Base name for output file
        
    Returns:
        Output file path
    """
    # Ensure outputs directory exists
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(output_dir / f"{base_name}_{timestamp}.mp4")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("REAL-TIME WORKOUT FORM ANALYZER")
    print("="*60)
    
    # Initialize pose detector
    print("\nInitializing pose detector...")
    try:
        pose_detector = PoseDetector(
            min_detection_confidence=args.confidence,
            min_tracking_confidence=args.confidence
        )
        print("✓ Pose detector initialized")
    except Exception as e:
        print(f"✗ Error initializing pose detector: {e}")
        return 1
    
    # Initialize exercise analyzer
    print(f"Loading exercise analyzer: {args.exercise}")
    try:
        if args.exercise == 'push-ups':
            analyzer = PushUpAnalyzer(pose_detector)
        elif args.exercise == 'bicep-curls':
            analyzer = BicepCurlAnalyzer(pose_detector)
        else:
            raise ValueError(f"Unknown exercise: {args.exercise}")
        print(f"✓ Exercise analyzer loaded: {args.exercise}")
    except Exception as e:
        print(f"✗ Error initializing exercise analyzer: {e}")
        return 1
    
    # Prepare output path
    if args.output:
        output_path = args.output
        # Ensure .mp4 extension
        if not output_path.lower().endswith('.mp4'):
            output_path += '.mp4'
    else:
        output_path = create_output_path(args.exercise.replace('-', '_'))
        print(f"Output file not specified. Will save to: {output_path}")
    
    # Initialize video processor
    print("\nInitializing video processor...")
    try:
        processor = VideoProcessor(analyzer, output_path=output_path)
        print("✓ Video processor initialized")
    except Exception as e:
        print(f"✗ Error initializing video processor: {e}")
        return 1
    
    # Process video
    print(f"\nStarting video processing from input: {args.input}")
    print(f"Display live video: {not args.no_display}")
    print("\nPress 'q' in the video window to stop processing.")
    print("-"*60)
    
    try:
        success = processor.process_video(
            args.input,
            display=not args.no_display
        )
        
        if success:
            print(f"\n✓ Processing completed successfully")
            if args.output:
                print(f"✓ Output saved to: {output_path}")
            return 0
        else:
            print(f"\n✗ Processing failed")
            return 1
    
    except Exception as e:
        print(f"\n✗ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
