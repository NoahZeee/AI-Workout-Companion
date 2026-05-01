"""
FeedbackGenerator: Handles real-time feedback display and frame annotation.
Formats feedback messages and draws them on video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class FeedbackGenerator:
    """
    Generates and displays real-time feedback on video frames.
    Handles text rendering, rep counter display, and form warnings.
    """
    
    # Font and text rendering
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    TEXT_COLOR = (255, 255, 255)  # White
    BG_COLOR = (0, 0, 0)  # Black background
    WARNING_COLOR = (0, 0, 255)  # Red
    SUCCESS_COLOR = (0, 255, 0)  # Green
    INFO_COLOR = (200, 200, 0)  # Cyan
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize feedback generator.
        
        Args:
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.message_history = []  # Keep track of recent messages for persistence
    
    def annotate_frame(self, frame: np.ndarray, rep_count: int, feedback_messages: List[str],
                      form_status: Dict, in_rep: bool = False) -> np.ndarray:
        """
        Annotate frame with feedback information.
        
        Args:
            frame: BGR image frame
            rep_count: Current rep count
            feedback_messages: List of feedback messages
            form_status: Dict with form metrics
            in_rep: Whether currently performing a rep
            
        Returns:
            Annotated frame
        """
        frame_annotated = frame.copy()
        
        # Draw rep counter prominently at top
        self._draw_rep_counter(frame_annotated, rep_count, in_rep)
        
        # Draw feedback messages
        self._draw_feedback_messages(frame_annotated, feedback_messages)
        
        # Draw form metrics
        self._draw_form_metrics(frame_annotated, form_status)
        
        return frame_annotated
    
    def _draw_rep_counter(self, frame: np.ndarray, rep_count: int, in_rep: bool):
        """
        Draw large rep counter at top of frame.
        
        Args:
            frame: Frame to annotate
            rep_count: Current rep count
            in_rep: Whether currently in a rep
        """
        text = f"REPS: {rep_count}"
        font_scale = 1.5
        font_thickness = 3
        
        # Get text size for background
        text_size = cv2.getTextSize(text, self.FONT, font_scale, font_thickness)[0]
        
        # Position at top center
        x = (self.frame_width - text_size[0]) // 2
        y = 50
        
        # Draw background rectangle
        bg_color = self.SUCCESS_COLOR if rep_count > 0 else self.INFO_COLOR
        cv2.rectangle(frame, (x - 10, y - text_size[1] - 10),
                     (x + text_size[0] + 10, y + 10), bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), self.FONT, font_scale,
                   (0, 0, 0), font_thickness)
        
        # Draw "IN PROGRESS" indicator if currently doing a rep
        if in_rep:
            status_text = "IN PROGRESS..."
            status_size = cv2.getTextSize(status_text, self.FONT, 0.7, 2)[0]
            status_x = (self.frame_width - status_size[0]) // 2
            status_y = y + 50
            cv2.rectangle(frame, (status_x - 5, status_y - status_size[1] - 5),
                         (status_x + status_size[0] + 5, status_y + 5),
                         self.WARNING_COLOR, -1)
            cv2.putText(frame, status_text, (status_x, status_y),
                       self.FONT, 0.7, (255, 255, 255), 2)
    
    def _draw_feedback_messages(self, frame: np.ndarray, feedback_messages: List[str]):
        """
        Draw feedback messages at top-left of frame.
        
        Args:
            frame: Frame to annotate
            feedback_messages: List of messages
        """
        if not feedback_messages:
            return
        
        y_offset = 120
        line_height = 30
        
        for message in feedback_messages[:5]:  # Limit to 5 messages
            # Determine color based on message content
            if "Rep" in message or "complete" in message:
                color = self.SUCCESS_COLOR
            elif "Go deeper" in message or "sagging" in message or "uneven" in message:
                color = self.WARNING_COLOR
            else:
                color = self.INFO_COLOR
            
            # Draw background for readability
            text_size = cv2.getTextSize(message, self.FONT, self.FONT_SCALE, 1)[0]
            cv2.rectangle(frame, (10, y_offset - text_size[1] - 5),
                         (20 + text_size[0], y_offset + 5), self.BG_COLOR, -1)
            
            # Draw text
            cv2.putText(frame, message, (15, y_offset), self.FONT,
                       self.FONT_SCALE, color, 1)
            
            y_offset += line_height
    
    def _draw_form_metrics(self, frame: np.ndarray, form_status: Dict):
        """
        Draw form metrics and angle information at bottom of frame.
        
        Args:
            frame: Frame to annotate
            form_status: Dict with form metrics
        """
        if not form_status:
            return
        
        metrics_text = []
        
        # Extract metrics
        if 'avg_elbow_angle' in form_status:
            avg_angle = form_status['avg_elbow_angle']
            metrics_text.append(f"Elbow Angle: {avg_angle:.1f}°")
        
        # Draw at bottom-right
        y_offset = self.frame_height - 20
        for i, text in enumerate(reversed(metrics_text)):
            text_size = cv2.getTextSize(text, self.FONT, 0.5, 1)[0]
            x = self.frame_width - text_size[0] - 10
            y = y_offset - (i * 25)
            
            # Semi-transparent background
            cv2.rectangle(frame, (x - 5, y - text_size[1] - 5),
                         (x + text_size[0] + 5, y + 5), self.BG_COLOR, -1)
            
            cv2.putText(frame, text, (x, y), self.FONT, 0.5, self.INFO_COLOR, 1)
    
    def create_summary_frame(self, final_rep_count: int, exercise_name: str,
                            session_duration: float) -> np.ndarray:
        """
        Create a summary frame showing final stats.
        
        Args:
            final_rep_count: Total reps completed
            exercise_name: Name of exercise
            session_duration: Duration of session in seconds
            
        Returns:
            Summary frame (BGR image)
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Title
        title = "WORKOUT COMPLETE"
        title_size = cv2.getTextSize(title, self.FONT, 2, 3)[0]
        title_x = (640 - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 80), self.FONT, 2,
                   self.SUCCESS_COLOR, 3)
        
        # Exercise name
        exercise_text = f"Exercise: {exercise_name}"
        exercise_size = cv2.getTextSize(exercise_text, self.FONT, 1.2, 2)[0]
        exercise_x = (640 - exercise_size[0]) // 2
        cv2.putText(frame, exercise_text, (exercise_x, 150), self.FONT, 1.2,
                   self.TEXT_COLOR, 2)
        
        # Final stats
        stats = [
            f"Total Reps: {final_rep_count}",
            f"Duration: {session_duration:.1f}s",
            f"Avg Time/Rep: {session_duration / max(final_rep_count, 1):.1f}s"
        ]
        
        y_offset = 240
        for stat in stats:
            stat_size = cv2.getTextSize(stat, self.FONT, 1.0, 2)[0]
            stat_x = (640 - stat_size[0]) // 2
            cv2.putText(frame, stat, (stat_x, y_offset), self.FONT, 1.0,
                       self.INFO_COLOR, 2)
            y_offset += 50
        
        # Footer
        footer = "Video saved successfully!"
        footer_size = cv2.getTextSize(footer, self.FONT, 0.8, 1)[0]
        footer_x = (640 - footer_size[0]) // 2
        cv2.putText(frame, footer, (footer_x, 420), self.FONT, 0.8,
                   self.SUCCESS_COLOR, 2)
        
        return frame
