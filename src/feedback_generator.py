"""
FeedbackGenerator: Modern, real-time feedback visualization for workout analysis.
Provides clean UI with visual form indicators and color-coded feedback.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class FeedbackGenerator:
    """
    Generates modern, real-time visual feedback on video frames.
    Features: rep counter, form metrics, color-coded indicators, visual guides.
    """
    
    # Colors (BGR)
    COLOR_GOOD = (0, 200, 50)        # Green - good form
    COLOR_WARNING = (0, 165, 255)    # Orange - needs attention
    COLOR_ERROR = (0, 0, 255)        # Red - form issue
    COLOR_INFO = (255, 200, 0)       # Cyan/Blue
    COLOR_NEUTRAL = (100, 100, 100)  # Gray
    COLOR_BG_DARK = (20, 20, 30)     # Dark blue-gray background
    COLOR_TEXT = (240, 240, 240)     # Off-white text
    
    # Fonts
    FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
    FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize feedback generator.
        
        Args:
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
        """
        self.width = frame_width
        self.height = frame_height
        self.frame_count = 0
    
    def annotate_frame(self, frame: np.ndarray, rep_count: int, 
                      feedback_messages: List[str], form_status: Dict, 
                      in_rep: bool = False) -> np.ndarray:
        """
        Annotate frame with modern feedback UI.
        
        Args:
            frame: BGR image frame
            rep_count: Current rep count
            feedback_messages: List of feedback messages
            form_status: Dict with form metrics
            in_rep: Whether currently performing a rep
            
        Returns:
            Annotated frame
        """
        self.frame_count += 1
        frame_out = frame.copy()
        
        # Draw semi-transparent overlay panel on right side (info panel)
        self._draw_info_panel(frame_out, rep_count, form_status, in_rep)
        
        # Draw primary rep counter (large, prominent)
        self._draw_rep_counter_large(frame_out, rep_count, in_rep)
        
        # Draw feedback messages (left side, with color coding)
        self._draw_feedback_messages(frame_out, feedback_messages)
        
        # Draw form indicators on frame (visual cues)
        self._draw_form_indicators(frame_out, form_status)
        
        return frame_out
    
    def _draw_rep_counter_large(self, frame: np.ndarray, rep_count: int, in_rep: bool):
        """
        Draw large, prominent rep counter at top center.
        
        Args:
            frame: Frame to draw on
            rep_count: Current rep count
            in_rep: Whether currently in a rep
        """
        # Main rep counter box
        box_width = 200
        box_height = 120
        box_x = (self.width - box_width) // 2
        box_y = 15
        
        # Background color based on state
        if in_rep:
            bg_color = self.COLOR_WARNING  # Orange when in rep
        elif rep_count > 0:
            bg_color = self.COLOR_GOOD     # Green when reps completed
        else:
            bg_color = self.COLOR_INFO     # Blue when idle
        
        # Draw background with rounded corners effect
        cv2.rectangle(frame, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     bg_color, -1)
        
        # Draw border
        cv2.rectangle(frame, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     self.COLOR_TEXT, 2)
        
        # Draw rep count (large text)
        rep_text = str(rep_count)
        text_size = cv2.getTextSize(rep_text, self.FONT_MAIN, 3.0, 3)[0]
        text_x = box_x + (box_width - text_size[0]) // 2
        text_y = box_y + box_height // 2 + text_size[1] // 2
        cv2.putText(frame, rep_text, (text_x, text_y), self.FONT_MAIN, 
                   3.0, self.COLOR_TEXT, 3)
        
        # Draw "REPS" label
        label_size = cv2.getTextSize("REPS", self.FONT_MONO, 0.7, 1)[0]
        label_x = box_x + (box_width - label_size[0]) // 2
        label_y = box_y + 25
        cv2.putText(frame, "REPS", (label_x, label_y), self.FONT_MONO,
                   0.7, self.COLOR_TEXT, 1)
        
        # Draw status indicator if in rep
        if in_rep:
            status_text = "IN PROGRESS"
            status_size = cv2.getTextSize(status_text, self.FONT_MONO, 0.6, 1)[0]
            status_x = (self.width - status_size[0]) // 2
            status_y = box_y + box_height + 25
            
            # Draw background for status
            cv2.rectangle(frame, (status_x - 5, status_y - status_size[1] - 3),
                         (status_x + status_size[0] + 5, status_y + 5),
                         self.COLOR_WARNING, -1)
            cv2.putText(frame, status_text, (status_x, status_y), self.FONT_MONO,
                       0.6, self.COLOR_TEXT, 1)
    
    def _draw_info_panel(self, frame: np.ndarray, rep_count: int, 
                        form_status: Dict, in_rep: bool):
        """
        Draw semi-transparent info panel on right side.
        
        Args:
            frame: Frame to draw on
            rep_count: Current rep count
            form_status: Form metrics dict
            in_rep: Whether currently in rep
        """
        # Panel dimensions
        panel_width = 280
        panel_height = 200
        panel_x = self.width - panel_width - 10
        panel_y = self.height - panel_height - 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     self.COLOR_INFO, 2)
        
        # Draw panel content
        content_x = panel_x + 15
        content_y = panel_y + 30
        line_height = 28
        
        # Title
        cv2.putText(frame, "FORM METRICS", (content_x, content_y),
                   self.FONT_MONO, 0.7, self.COLOR_INFO, 1)
        content_y += line_height + 5
        
        # Elbow angle
        if 'avg_elbow_angle' in form_status:
            angle = form_status['avg_elbow_angle']
            angle_status = self._get_angle_status(angle, in_rep)
            angle_color = self._get_status_color(angle_status)
            
            text = f"Elbow: {angle:.0f}°"
            cv2.putText(frame, text, (content_x, content_y),
                       self.FONT_MONO, 0.65, angle_color, 1)
            content_y += line_height
        
        # Visible side
        if 'visible_side' in form_status:
            side = form_status['visible_side']
            side_text = f"Side: {side.upper()}"
            cv2.putText(frame, side_text, (content_x, content_y),
                       self.FONT_MONO, 0.65, self.COLOR_INFO, 1)
            content_y += line_height
        
        # Form status indicator
        form_issues = form_status.get('form_issues', [])
        if not form_issues:
            status_text = "✓ Good Form"
            status_color = self.COLOR_GOOD
        else:
            status_text = f"⚠ {len(form_issues)} issue(s)"
            status_color = self.COLOR_ERROR
        
        cv2.putText(frame, status_text, (content_x, content_y),
                   self.FONT_MONO, 0.65, status_color, 1)
    
    def _draw_feedback_messages(self, frame: np.ndarray, 
                               feedback_messages: List[str]):
        """
        Draw feedback messages on left side with color coding.
        
        Args:
            frame: Frame to draw on
            feedback_messages: List of feedback messages
        """
        if not feedback_messages:
            return
        
        # Limit to 4 messages and reverse to show most recent at top
        messages = feedback_messages[:4]
        
        start_y = 200
        line_spacing = 35
        
        for i, msg in enumerate(messages):
            y = start_y + (i * line_spacing)
            
            # Determine color based on message content
            if "Rep" in msg or "complete" in msg:
                color = self.COLOR_GOOD
            elif "deeper" in msg or "sagging" in msg or "uneven" in msg or "Ensure" in msg:
                color = self.COLOR_ERROR
            else:
                color = self.COLOR_WARNING
            
            # Draw message background
            text_size = cv2.getTextSize(msg, self.FONT_MONO, 0.65, 1)[0]
            bg_x1, bg_y1 = 10, y - text_size[1] - 5
            bg_x2, bg_y2 = 20 + text_size[0], y + 5
            
            # Slight background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                         self.COLOR_BG_DARK, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                         color, 1)
            
            # Draw text
            cv2.putText(frame, msg, (15, y), self.FONT_MONO, 0.65, color, 1)
    
    def _draw_form_indicators(self, frame: np.ndarray, form_status: Dict):
        """
        Draw visual form indicators on the frame.
        
        Args:
            frame: Frame to draw on
            form_status: Form metrics dict
        """
        # Draw form status in top-right
        form_issues = form_status.get('form_issues', [])
        
        if form_issues:
            # Draw warning indicator
            indicator_x = self.width - 40
            indicator_y = 30
            indicator_size = 12
            
            # Pulsing effect using frame count
            pulse = 3 + 2 * abs(np.sin(self.frame_count * 0.1))
            
            cv2.circle(frame, (indicator_x, indicator_y), int(indicator_size + pulse),
                      self.COLOR_ERROR, 2)
            cv2.circle(frame, (indicator_x, indicator_y), int(indicator_size),
                      self.COLOR_ERROR, -1)
    
    def _get_angle_status(self, angle: float, in_rep: bool) -> str:
        """
        Determine status based on elbow angle.
        
        Args:
            angle: Elbow angle in degrees
            in_rep: Whether currently in a rep
            
        Returns:
            Status string: 'good', 'warning', or 'error'
        """
        if in_rep:
            # During rep, angles < 100 are good (bent elbows)
            if angle < 100:
                return 'good'
            elif angle < 120:
                return 'warning'
            else:
                return 'error'
        else:
            # At rest, angles > 150 are good (straight arms)
            if angle > 150:
                return 'good'
            else:
                return 'warning'
    
    def _get_status_color(self, status: str) -> Tuple[int, int, int]:
        """
        Get color for status.
        
        Args:
            status: Status string ('good', 'warning', 'error')
            
        Returns:
            BGR color tuple
        """
        if status == 'good':
            return self.COLOR_GOOD
        elif status == 'warning':
            return self.COLOR_WARNING
        else:
            return self.COLOR_ERROR
    
    def create_summary_frame(self, final_rep_count: int, exercise_name: str,
                            session_duration: float) -> np.ndarray:
        """
        Create modern summary frame.
        
        Args:
            final_rep_count: Total reps completed
            exercise_name: Name of exercise
            session_duration: Duration in seconds
            
        Returns:
            Summary frame (BGR image)
        """
        # Create frame with gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(self.height):
            intensity = int(50 + (y / self.height) * 30)
            frame[y, :] = [intensity, intensity, intensity + 20]
        
        # Title
        title = "WORKOUT COMPLETE"
        title_size = cv2.getTextSize(title, self.FONT_MAIN, 2.2, 2)[0]
        title_x = (self.width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 80), self.FONT_MAIN, 2.2,
                   self.COLOR_GOOD, 2)
        
        # Final rep count (large)
        reps_text = f"{final_rep_count} REPS"
        reps_size = cv2.getTextSize(reps_text, self.FONT_MAIN, 2.5, 2)[0]
        reps_x = (self.width - reps_size[0]) // 2
        cv2.putText(frame, reps_text, (reps_x, 180), self.FONT_MAIN, 2.5,
                   self.COLOR_INFO, 2)
        
        # Exercise name
        exercise_text = f"{exercise_name}"
        exercise_size = cv2.getTextSize(exercise_text, self.FONT_MAIN, 1.5, 2)[0]
        exercise_x = (self.width - exercise_size[0]) // 2
        cv2.putText(frame, exercise_text, (exercise_x, 250), self.FONT_MAIN, 1.5,
                   self.COLOR_TEXT, 1)
        
        # Stats
        duration_text = f"Duration: {session_duration:.1f}s"
        if final_rep_count > 0:
            avg_text = f"Avg Time: {session_duration / final_rep_count:.1f}s/rep"
        else:
            avg_text = "Avg Time: N/A"
        
        stats_size = cv2.getTextSize(duration_text, self.FONT_MONO, 1.0, 1)[0]
        stats_x = (self.width - stats_size[0]) // 2
        
        cv2.putText(frame, duration_text, (stats_x, 320), self.FONT_MONO, 1.0,
                   self.COLOR_TEXT, 1)
        
        avg_size = cv2.getTextSize(avg_text, self.FONT_MONO, 1.0, 1)[0]
        avg_x = (self.width - avg_size[0]) // 2
        cv2.putText(frame, avg_text, (avg_x, 360), self.FONT_MONO, 1.0,
                   self.COLOR_TEXT, 1)
        
        # Footer
        footer = "Analysis Complete"
        footer_size = cv2.getTextSize(footer, self.FONT_MONO, 0.9, 1)[0]
        footer_x = (self.width - footer_size[0]) // 2
        cv2.putText(frame, footer, (footer_x, self.height - 40), self.FONT_MONO, 0.9,
                   self.COLOR_GOOD, 1)
        
        return frame

