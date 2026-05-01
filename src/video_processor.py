"""
VideoProcessor: Handles video input/output and main processing loop.
Integrates pose detection, exercise analysis, and feedback generation.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple
from pose_detector import PoseDetector
from exercise_analyzer import ExerciseAnalyzer
from feedback_generator import FeedbackGenerator


class VideoProcessor:
    """
    Processes video frames from webcam or file, applies pose detection and
    exercise analysis, and saves annotated output video.
    """
    
    def __init__(self, exercise_analyzer: ExerciseAnalyzer, output_path: Optional[str] = None):
        """
        Initialize video processor.
        
        Args:
            exercise_analyzer: ExerciseAnalyzer instance (e.g., PushUpAnalyzer)
            output_path: Path to save output video. If None, video won't be saved.
        """
        self.analyzer = exercise_analyzer
        self.pose_detector = exercise_analyzer.pose_detector
        self.feedback_generator = None
        self.output_path = output_path
        self.video_writer = None
        self.start_time = None
        self.frame_count = 0
    
    def process_video(self, input_source: str, display: bool = True) -> bool:
        """
        Process video from webcam or file.
        
        Args:
            input_source: Either:
                - "0" or int: Webcam device ID (default 0)
                - String path: Path to .mp4 or video file
            display: Whether to display live video window
            
        Returns:
            True if processing completed successfully, False otherwise
        """
        # Open video source
        cap = self._open_video_source(input_source)
        if cap is None:
            print(f"Error: Could not open video source: {input_source}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Source: {input_source}")
        print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
        if total_frames > 0:
            print(f"Total Frames: {total_frames} (~{total_frames/fps:.1f}s)")
        
        # Initialize feedback generator and video writer
        self.feedback_generator = FeedbackGenerator(width, height)
        if self.output_path:
            self.video_writer = self._init_video_writer(width, height, fps)
            if self.video_writer is None:
                print("Error: Could not initialize video writer")
                cap.release()
                return False
        
        self.start_time = time.time()
        self.frame_count = 0
        success = True
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("\nEnd of video reached or error reading frame")
                    break
                
                # Process frame
                self.frame_count += 1
                annotated_frame = self._process_frame(frame)
                
                # Display live video if requested
                if display:
                    cv2.imshow("Workout Analyzer", annotated_frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser interrupted processing")
                        break
                
                # Write to output video
                if self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                # Print progress every 30 frames
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps_actual = self.frame_count / elapsed
                    print(f"Frame {self.frame_count}: {fps_actual:.1f} FPS, "
                          f"Reps: {self.analyzer.rep_count}")
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected")
        
        finally:
            # Cleanup
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print summary
        elapsed_time = time.time() - self.start_time
        self._print_summary(elapsed_time)
        
        # Save summary frame if output video was created
        if self.video_writer and self.output_path:
            self._save_summary_frame(elapsed_time)
        
        return success
    
    def _open_video_source(self, input_source: str) -> Optional[cv2.VideoCapture]:
        """
        Open video source (webcam or file).
        
        Args:
            input_source: Webcam ID (int/str) or file path
            
        Returns:
            cv2.VideoCapture object or None
        """
        try:
            # Try to interpret as webcam ID
            if input_source.isdigit():
                device_id = int(input_source)
                print(f"Opening webcam (device {device_id})...")
                cap = cv2.VideoCapture(device_id)
                
                # Set webcam to 720p (1280x720) for better quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            else:
                # Treat as file path
                if not Path(input_source).exists():
                    print(f"Error: File not found: {input_source}")
                    return None
                print(f"Opening video file: {input_source}")
                cap = cv2.VideoCapture(input_source)
            
            # Check if opened successfully
            if not cap.isOpened():
                return None
            
            return cap
        except Exception as e:
            print(f"Error opening video source: {e}")
            return None
    
    def _init_video_writer(self, width: int, height: int, fps: float) -> Optional[cv2.VideoWriter]:
        """
        Initialize video writer for output file.
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            
        Returns:
            cv2.VideoWriter object or None
        """
        try:
            # Use H.264 codec (common, good compression)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Ensure output directory exists
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Output video will be saved to: {self.output_path}")
            
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                print("Error: Could not open video writer")
                return None
            
            return writer
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            return None
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect pose, analyze exercise, generate feedback.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Annotated frame
        """
        # Detect pose
        landmarks, visibility = self.pose_detector.detect_pose(frame)
        
        annotated_frame = frame.copy()
        
        # If no pose detected, show message
        if landmarks is None or visibility is None:
            cv2.putText(annotated_frame, "No pose detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame
        
        # Draw pose skeleton
        annotated_frame = self.pose_detector.draw_pose(
            annotated_frame, landmarks, visibility, confidence_threshold=0.5
        )
        
        # Analyze exercise
        analysis_result = self.analyzer.analyze(landmarks, visibility)
        
        # Generate and annotate feedback
        annotated_frame = self.feedback_generator.annotate_frame(
            annotated_frame,
            rep_count=analysis_result['rep_count'],
            feedback_messages=analysis_result['feedback'],
            form_status=analysis_result['form_status'],
            in_rep=analysis_result['in_rep']
        )
        
        return annotated_frame
    
    def _save_summary_frame(self, elapsed_time: float):
        """
        Save a summary frame at the end of the output video.
        
        Args:
            elapsed_time: Total processing time in seconds
        """
        try:
            summary_frame = self.feedback_generator.create_summary_frame(
                final_rep_count=self.analyzer.rep_count,
                exercise_name=self.analyzer.__class__.__name__.replace("Analyzer", ""),
                session_duration=elapsed_time
            )
            
            # Write summary frame 10 times (1 second at 10 fps)
            for _ in range(10):
                self.video_writer.write(summary_frame)
            
            print("Summary frame added to output video")
        except Exception as e:
            print(f"Error saving summary frame: {e}")
    
    def _print_summary(self, elapsed_time: float):
        """
        Print workout session summary.
        
        Args:
            elapsed_time: Total session time in seconds
        """
        print("\n" + "="*50)
        print("WORKOUT SESSION COMPLETE")
        print("="*50)
        print(f"Exercise: {self.analyzer.__class__.__name__.replace('Analyzer', '')}")
        print(f"Total Reps: {self.analyzer.rep_count}")
        print(f"Duration: {elapsed_time:.1f}s")
        print(f"Frames Processed: {self.frame_count}")
        print(f"Avg FPS: {self.frame_count / elapsed_time:.1f}")
        if self.analyzer.rep_count > 0:
            print(f"Avg Time/Rep: {elapsed_time / self.analyzer.rep_count:.1f}s")
        if self.output_path:
            print(f"Output Video: {self.output_path}")
        print("="*50)
