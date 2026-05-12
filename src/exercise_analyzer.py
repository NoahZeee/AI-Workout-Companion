"""
ExerciseAnalyzer: Exercise-specific analysis for rep counting and form feedback.
Optimized for single-side (side-view) detection where only one arm is typically visible.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
from pose_detector import PoseDetector


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


class PushUpAnalyzer(ExerciseAnalyzer):
    """
    Push-up analyzer optimized for side-view camera angles.
    Works with only one arm visible (left or right).
    
    Rep detection: DOWN (elbow < 100°) -> UP (elbow > 150°) = 1 rep
    """
    
    ELBOW_DOWN_THRESHOLD = 100   # Bottom of push-up
    ELBOW_UP_THRESHOLD = 150     # Top of push-up
    MIN_DEPTH_ANGLE = 90         # Minimum elbow bend for valid rep
    
    def __init__(self, pose_detector: PoseDetector):
        """Initialize push-up analyzer."""
        super().__init__(pose_detector)
        self.rep_phase = "up"
        self.max_depth_angle = 180
        self.active_side = None
    
    def get_required_landmarks(self) -> List[str]:
        """Required landmarks: shoulders and one visible arm."""
        return [
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST"
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
        
        # Track minimum angle in current rep
        if elbow_angle < self.max_depth_angle:
            self.max_depth_angle = elbow_angle
        
        # Count reps
        self._detect_rep(elbow_angle)
        
        # Validate form
        form_issues = self._validate_form(landmarks, elbow_angle)
        
        return {
            'rep_count': self.rep_count,
            'in_rep': self.in_rep,
            'feedback': self.feedback_messages,
            'form_status': {
                'elbow_angle': elbow_angle,
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
    
    def _detect_rep(self, elbow_angle: float):
        """Detect and count reps using state machine."""
        if self.rep_phase == "up" and elbow_angle < self.ELBOW_DOWN_THRESHOLD:
            self.rep_phase = "down"
            self.in_rep = True
            self.max_depth_angle = elbow_angle
            self.feedback_messages.append("Going down...")
        
        elif self.rep_phase == "down" and elbow_angle > self.ELBOW_UP_THRESHOLD:
            # Check if rep reached adequate depth before counting
            if self.max_depth_angle > self.MIN_DEPTH_ANGLE:
                self.feedback_messages.append("Go deeper!")
            
            self.rep_phase = "up"
            self.rep_count += 1
            self.in_rep = False
            self.max_depth_angle = 180
            self.feedback_messages.append(f"Rep {self.rep_count} complete!")
    
    def _validate_form(self, landmarks: np.ndarray, elbow_angle: float) -> List[str]:
        """
        Validate push-up form. Side-view optimized (no symmetry checks).
        Depth validation is checked at rep completion in _detect_rep().
        """
        return []


class BicepCurlAnalyzer(ExerciseAnalyzer):
    """
    Bicep curl analyzer for side-view angles.
    Works with only one arm visible.
    
    Rep detection: DOWN (extended) -> UP (flexed) -> DOWN = 1 rep
    """
    
    ELBOW_DOWN_THRESHOLD = 160  # Arm extended
    ELBOW_UP_THRESHOLD = 60     # Arm flexed
    
    def __init__(self, pose_detector: PoseDetector):
        """Initialize bicep curl analyzer."""
        super().__init__(pose_detector)
        self.rep_phase = "down"
        self.active_side = None
    
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
                'feedback': ["Need arm visible"],
                'form_status': {}
            }
        
        # Analyze visible arm
        elbow_angle = self._calculate_elbow_angle(landmarks, self.active_side)
        
        # Count reps
        self._detect_rep(elbow_angle)
        
        # Form validation
        form_issues = self._validate_form(elbow_angle)
        
        return {
            'rep_count': self.rep_count,
            'in_rep': self.in_rep,
            'feedback': self.feedback_messages,
            'form_status': {
                'elbow_angle': elbow_angle,
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
    
    def _detect_rep(self, elbow_angle: float):
        """Detect reps: flex and extend."""
        if self.rep_phase == "down" and elbow_angle < self.ELBOW_UP_THRESHOLD:
            self.rep_phase = "up"
            self.in_rep = True
            self.feedback_messages.append("Flexing...")
        
        elif self.rep_phase == "up" and elbow_angle > self.ELBOW_DOWN_THRESHOLD:
            self.rep_phase = "down"
            self.rep_count += 1
            self.in_rep = False
            self.feedback_messages.append(f"Rep {self.rep_count} complete!")
    
    def _validate_form(self, elbow_angle: float) -> List[str]:
        """Validate bicep curl form."""
        issues = []
        
        # In side-view, we can't check symmetry, so just check range
        if self.in_rep and elbow_angle > 100:
            issues.append("Flex more")
        
        self.feedback_messages.extend(issues)
        return issues
