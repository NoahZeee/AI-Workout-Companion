"""
create_test_video.py: Generate a test video with simple frames for testing the workout analyzer.
"""

import cv2
import numpy as np
from pathlib import Path


def create_test_video(output_path: str = "test_video.mp4", duration_seconds: float = 5, fps: int = 30):
    """
    Create a simple test video with moving rectangles.
    
    Args:
        output_path: Path to save the test video
        duration_seconds: Duration of test video in seconds
        fps: Frames per second
    """
    frame_width = 640
    frame_height = 480
    total_frames = int(duration_seconds * fps)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Creating test video: {output_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}, Duration: {duration_seconds}s, Frames: {total_frames}")
    
    try:
        for frame_num in range(total_frames):
            # Create a frame with a simple moving rectangle
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Add some text
            text = f"Test Frame {frame_num + 1}/{total_frames}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw a moving rectangle
            x = 100 + int(300 * np.sin(2 * np.pi * frame_num / total_frames))
            y = 200
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 0, 255), -1)
            
            # Add frame info
            cv2.putText(frame, f"FPS: {fps}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            out.write(frame)
            
            if (frame_num + 1) % 10 == 0:
                print(f"  Frame {frame_num + 1}/{total_frames}")
    
    finally:
        out.release()
    
    print(f"✓ Test video created: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")


if __name__ == '__main__':
    create_test_video("test_video.mp4", duration_seconds=5, fps=30)
