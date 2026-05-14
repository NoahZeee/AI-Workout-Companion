"""
ExerciseAnalyzer: Exercise-specific analysis for rep counting and form feedback.
Optimized for single-side (side-view) detection where only one arm is typically visible.

STRUCTURE:
  - ExerciseAnalyzer: Abstract base class for all exercises
  - PushUpAnalyzer: Push-up rep counting and form validation (depth + hip alignment)
  - BicepCurlAnalyzer: Bicep curl rep counting and form validation (flex + elbow alignment)

To add a new exercise:
  1. Subclass ExerciseAnalyzer
  2. Implement analyze() and get_required_landmarks()
  3. Register in src/main.py
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
from pose_detector import PoseDetector


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class ExerciseAnalyzer(ABC):
    """
    Abstract base class for exercise-specific analysis.
    Subclasses implement rep counting and form validation logic.
    """
    
    def __init__(self, pose_detector: PoseDetector):
        """
        Initialize exercise analyzer.
        
        Args:
            pose_detector: PoseDetector instance for accessing landmarks and angles
        """
        self.pose_detector = pose_detector
        self.rep_count = 0
        self.in_rep = False
        self.feedback_messages = []
    
    @abstractmethod
    def analyze(self, landmarks: np.ndarray, visibility: np.ndarray) -> Dict:
        """
        Analyze current frame for rep counting and form feedback.
        
        Args:
            landmarks: Nx3 array of pose landmarks
            visibility: N array of visibility scores
            
        Returns:
            Dict with rep count, feedback, and form metrics
        """
        pass
    
    @abstractmethod
    def get_required_landmarks(self) -> List[str]:
        """Get list of landmarks required for this exercise."""
        pass
    
    def reset(self):
        """Reset rep counter and state."""
        self.rep_count = 0
        self.in_rep = False
        self.feedback_messages = []


# ============================================================================
# PUSH-UP ANALYZER
# ============================================================================

class PushUpAnalyzer(ExerciseAnalyzer):
    """
    Push-up analyzer optimized for side-view camera angles.
    Works with only one arm visible (left or right).
    
    Rep detection: DOWN (elbow < 100°) -> UP (elbow > 150°) = 1 rep
    """
    
    ELBOW_DOWN_THRESHOLD = 100   # Bottom of push-up
    ELBOW_UP_THRESHOLD = 150     # Top of push-up
    MIN_DEPTH_ANGLE = 90         # Minimum elbow bend for valid rep
    HIP_SAG_TOLERANCE = 0.50     # Max body line deviation from straight (0-1 scale, ~18°)
    
    def __init__(self, pose_detector: PoseDetector):
        """Initialize push-up analyzer."""
        super().__init__(pose_detector)
        self.rep_phase = "up"
        self.max_depth_angle = 180
        self.min_hip_alignment_error = float('inf')  # Track best alignment during rep
        self.active_side = None
        self.hips_out_of_alignment = False  # Track persistent hip alignment state
    
    def get_required_landmarks(self) -> List[str]:
        """Required landmarks: shoulders, one visible arm, and hips for alignment check."""
        return [
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_HIP", "RIGHT_HIP"
        ]
    
    def analyze(self, landmarks: np.ndarray, visibility: np.ndarray) -> Dict:
        """
        Analyze push-up from side-view perspective.
        Detects which side is visible and analyzes only that side.
        """
        self.feedback_messages = []
        
        # Detect which side is more visible first
        visible_side = self.pose_detector.detect_visible_side(visibility)
        self.active_side = visible_side if visible_side != 'both' else 'right'
        
        # Validate only the visible side's landmarks + one shoulder
        if self.active_side == "left":
            required = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"]
        else:
            required = ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"]
        
        if not self.pose_detector.is_pose_valid(visibility, required, 0.1):
            return {
                'rep_count': self.rep_count,
                'in_rep': False,
                'feedback': ["Adjust view: arm not visible"],
                'form_status': {}
            }
        
        # Analyze visible arm
        elbow_angle = self._calculate_elbow_angle(landmarks, self.active_side)
        
        # Track minimum angle in current rep
        if elbow_angle < self.max_depth_angle:
            self.max_depth_angle = elbow_angle
        
        # Check hip alignment continuously
        current_alignment_error = self._calculate_hip_alignment_error(landmarks)
        self.hips_out_of_alignment = current_alignment_error > self.HIP_SAG_TOLERANCE
        
        # Track hip-shoulder alignment during rep (for validation at completion)
        if self.rep_phase == "down":
            if current_alignment_error < self.min_hip_alignment_error:
                self.min_hip_alignment_error = current_alignment_error
        
        # Add hip alignment feedback (positive or negative)
        if self.hips_out_of_alignment:
            self.feedback_messages.append("Keep hips aligned!")
        else:
            self.feedback_messages.append("Hips aligned")
        
        # Count reps
        self._detect_rep(landmarks, elbow_angle)
        
        # Validate form
        form_issues = self._validate_form(landmarks, elbow_angle)
        
        return {
            'rep_count': self.rep_count,
            'in_rep': self.in_rep,
            'feedback': self.feedback_messages,
            'form_status': {
                'avg_elbow_angle': elbow_angle,
                'visible_side': self.active_side,
                'form_issues': form_issues
            }
        }
    
    def _calculate_elbow_angle(self, landmarks: np.ndarray, side: str) -> float:
        """Calculate elbow angle for the visible arm."""
        if side == "left":
            shoulder = self.pose_detector.get_landmark(landmarks, "LEFT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "LEFT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "LEFT_WRIST")
        else:
            shoulder = self.pose_detector.get_landmark(landmarks, "RIGHT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "RIGHT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "RIGHT_WRIST")
        
        return self.pose_detector.calculate_angle(shoulder, elbow, wrist)
    
    def _calculate_hip_alignment_error(self, landmarks: np.ndarray) -> float:
        """
        Calculate body line alignment error.
        For proper push-up form, body should form straight line: shoulder-hip-knee.
        Uses knee instead of ankle (more visible during push-ups).
        Returns deviation from 180° (straight line).
        """
        # Get average positions for symmetrical landmarks
        left_shoulder = self.pose_detector.get_landmark(landmarks, "LEFT_SHOULDER")
        right_shoulder = self.pose_detector.get_landmark(landmarks, "RIGHT_SHOULDER")
        left_hip = self.pose_detector.get_landmark(landmarks, "LEFT_HIP")
        right_hip = self.pose_detector.get_landmark(landmarks, "RIGHT_HIP")
        left_knee = self.pose_detector.get_landmark(landmarks, "LEFT_KNEE")
        right_knee = self.pose_detector.get_landmark(landmarks, "RIGHT_KNEE")
        
        # Average the bilateral landmarks
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        avg_hip = (left_hip + right_hip) / 2
        avg_knee = (left_knee + right_knee) / 2
        
        # Calculate angle formed by shoulder-hip-knee
        # A straight body line = 180°
        body_line_angle = self.pose_detector.calculate_angle(avg_shoulder, avg_hip, avg_knee)
        
        # Calculate deviation from 180° (perfect straight line)
        angle_error = abs(180 - body_line_angle)
        
        # Normalize to 0-1 scale (0-45° error = acceptable)
        # More lenient tolerance since knee is further from hip than ankle
        # Clamp to [0, 1]
        normalized_error = min(angle_error / 45.0, 1.0)
        
        return normalized_error
    
    def _detect_rep(self, landmarks: np.ndarray, elbow_angle: float):
        """Detect and count reps using state machine with form validation."""
        if self.rep_phase == "up" and elbow_angle < self.ELBOW_DOWN_THRESHOLD:
            self.rep_phase = "down"
            self.in_rep = True
            self.max_depth_angle = elbow_angle
            self.min_hip_alignment_error = float('inf')
            self.feedback_messages.append("Going down...")
        
        elif self.rep_phase == "down" and elbow_angle > self.ELBOW_UP_THRESHOLD:
            # Validate depth
            depth_valid = self.max_depth_angle <= self.MIN_DEPTH_ANGLE
            
            # Validate alignment
            alignment_valid = self.min_hip_alignment_error <= self.HIP_SAG_TOLERANCE
            
            # Only count if both depth and alignment are good
            if depth_valid and alignment_valid:
                self.rep_phase = "up"
                self.rep_count += 1
                self.in_rep = False
                self.max_depth_angle = 180
                self.min_hip_alignment_error = float('inf')
                self.feedback_messages.append(f"Rep {self.rep_count} complete!")
            else:
                # Failed rep - provide feedback without counting
                if not depth_valid:
                    self.feedback_messages.append("Go deeper!")
                if not alignment_valid:
                    self.feedback_messages.append("Keep hips aligned!")
                
                # Reset for next attempt
                self.rep_phase = "up"
                self.in_rep = False
                self.max_depth_angle = 180
                self.min_hip_alignment_error = float('inf')
    
    def _validate_form(self, landmarks: np.ndarray, elbow_angle: float) -> List[str]:
        """
        Validate push-up form. Side-view optimized (no symmetry checks).
        Depth validation is checked at rep completion in _detect_rep().
        """
        return []



# ============================================================================
# BICEP CURL ANALYZER
# ============================================================================

class BicepCurlAnalyzer(ExerciseAnalyzer):
    """
    Bicep curl analyzer for side-view angles.
    Works with only one arm visible.
    
    Rep detection: DOWN (extended) -> UP (flexed) -> DOWN = 1 rep
    Form criteria:
      - Elbow flexion: <70 degrees
      - Elbow alignment: elbow stays directly below shoulder (vertical)
    """
    
    ELBOW_DOWN_THRESHOLD = 140  # Arm extended (relaxed from 160)
    ELBOW_UP_THRESHOLD = 80     # Arm flexed (relaxed from 70)
    MIN_FLEX_ANGLE = 80         # Minimum elbow bend for valid rep (relaxed from 70)
    ELBOW_ALIGNMENT_TOLERANCE = 0.60  # Max deviation from vertical (relaxed from 0.45, ~27° max)
    
    def __init__(self, pose_detector: PoseDetector):
        """Initialize bicep curl analyzer."""
        super().__init__(pose_detector)
        self.rep_phase = "down"
        self.active_side = None
        self.max_flex_angle = 180  # Track best flex during rep
        self.min_alignment_error = float('inf')  # Track best alignment during rep
        self.elbow_out_of_alignment = False
    
    def get_required_landmarks(self) -> List[str]:
        """Required: shoulder, elbow, wrist (any side)."""
        return [
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST"
        ]
    
    def analyze(self, landmarks: np.ndarray, visibility: np.ndarray) -> Dict:
        """Analyze bicep curl from whichever side is visible."""
        self.feedback_messages = []
        
        # Detect which side is visible first
        visible_side = self.pose_detector.detect_visible_side(visibility)
        self.active_side = visible_side if visible_side != 'both' else 'right'
        
        # Validate only the visible side's landmarks
        if self.active_side == "left":
            required = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
        else:
            required = ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
        
        if not self.pose_detector.is_pose_valid(visibility, required, 0.1):
            return {
                'rep_count': self.rep_count,
                'in_rep': False,
                'feedback': ["Adjust view: arm not visible"],
                'form_status': {}
            }
        
        # Analyze visible arm
        elbow_angle = self._calculate_elbow_angle(landmarks, self.active_side)
        
        # Track best flex during rep
        if elbow_angle < self.max_flex_angle:
            self.max_flex_angle = elbow_angle
        
        # Check elbow alignment continuously
        current_alignment_error = self._calculate_elbow_alignment_error(landmarks, self.active_side)
        self.elbow_out_of_alignment = current_alignment_error > self.ELBOW_ALIGNMENT_TOLERANCE
        
        # Track best alignment during rep (for validation at completion)
        if self.rep_phase == "up":
            if current_alignment_error < self.min_alignment_error:
                self.min_alignment_error = current_alignment_error
        
        # Add elbow alignment feedback
        if self.elbow_out_of_alignment:
            self.feedback_messages.append("Keep elbow still!")

        
        # Count reps
        self._detect_rep(elbow_angle)
        
        return {
            'rep_count': self.rep_count,
            'in_rep': self.in_rep,
            'feedback': self.feedback_messages,
            'form_status': {
                'avg_elbow_angle': elbow_angle,
                'visible_side': self.active_side,
                'alignment_error': current_alignment_error
            }
        }
    
    def _calculate_elbow_angle(self, landmarks: np.ndarray, side: str) -> float:
        """Calculate elbow angle for the visible arm."""
        if side == "left":
            shoulder = self.pose_detector.get_landmark(landmarks, "LEFT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "LEFT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "LEFT_WRIST")
        else:
            shoulder = self.pose_detector.get_landmark(landmarks, "RIGHT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "RIGHT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "RIGHT_WRIST")
        
        return self.pose_detector.calculate_angle(shoulder, elbow, wrist)
    
    def _calculate_elbow_alignment_error(self, landmarks: np.ndarray, side: str) -> float:
        """
        Calculate how well the elbow stays directly below the shoulder.
        For proper bicep curl form, shoulder-to-elbow line should be vertical.
        Returns deviation from vertical (0-1 scale).
        """
        if side == "left":
            shoulder = self.pose_detector.get_landmark(landmarks, "LEFT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "LEFT_ELBOW")
        else:
            shoulder = self.pose_detector.get_landmark(landmarks, "RIGHT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "RIGHT_ELBOW")
        
        # Calculate vector from shoulder to elbow
        shoulder_to_elbow = elbow - shoulder
        
        # For vertical alignment, x-component should be 0 (no horizontal drift)
        # Calculate angle from vertical (0, 1) direction
        horizontal_drift = abs(shoulder_to_elbow[0])
        vertical_distance = abs(shoulder_to_elbow[1])
        
        # Avoid division by zero
        if vertical_distance < 0.01:
            return 1.0
        
        # Calculate angle from vertical using tangent
        # angle_from_vertical = atan(horizontal_drift / vertical_distance)
        # Normalized to 0-1 (0-45° deviation = acceptable)
        angle_from_vertical = np.arctan2(horizontal_drift, vertical_distance) * 180 / np.pi
        normalized_error = min(angle_from_vertical / 45.0, 1.0)
        
        return normalized_error
    
    def _detect_rep(self, elbow_angle: float):
        """
        Detect reps with form validation.
        Rep must meet: minimum flex angle AND elbow alignment.
        """
        if self.rep_phase == "down" and elbow_angle < self.ELBOW_UP_THRESHOLD:
            self.rep_phase = "up"
            self.in_rep = True
            self.max_flex_angle = elbow_angle
            self.min_alignment_error = float('inf')
            self.feedback_messages.append("Curling up...")
        
        elif self.rep_phase == "up" and elbow_angle > self.ELBOW_DOWN_THRESHOLD:
            # Validate flex depth
            flex_valid = self.max_flex_angle <= self.MIN_FLEX_ANGLE
            
            # Validate alignment
            alignment_valid = self.min_alignment_error <= self.ELBOW_ALIGNMENT_TOLERANCE
            
            # Only count if both are valid
            if flex_valid and alignment_valid:
                self.rep_phase = "down"
                self.rep_count += 1
                self.in_rep = False
                self.max_flex_angle = 180
                self.min_alignment_error = float('inf')
                self.feedback_messages.append(f"Rep {self.rep_count} complete!")
            else:
                # Failed rep - provide feedback
                if not flex_valid:
                    self.feedback_messages.append("Curl more!")
                if not alignment_valid:
                    self.feedback_messages.append("Keep elbow still!")
                
                # Reset for next attempt
                self.rep_phase = "down"
                self.in_rep = False
                self.max_flex_angle = 180
                self.min_alignment_error = float('inf')
