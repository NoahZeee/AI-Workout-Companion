"""
ExerciseAnalyzer: Base class and concrete implementations for exercise-specific analysis.
Handles rep counting and form feedback for different workout exercises.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple
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
        self.in_rep = False  # Track if currently in a rep cycle
        self.feedback_messages = []
    
    @abstractmethod
    def analyze(self, landmarks: np.ndarray, visibility: np.ndarray) -> Dict:
        """
        Analyze current frame for rep counting and form feedback.
        
        Args:
            landmarks: Nx3 array of pose landmarks
            visibility: N array of visibility scores
            
        Returns:
            Dict with keys:
            - 'rep_count': Current total reps
            - 'in_rep': Whether currently performing a rep
            - 'feedback': List of feedback messages
            - 'form_status': Dict of form metrics
        """
        pass
    
    @abstractmethod
    def get_required_landmarks(self) -> List[str]:
        """
        Get list of landmarks required for this exercise.
        
        Returns:
            List of landmark names
        """
        pass
    
    def reset(self):
        """Reset rep counter and state."""
        self.rep_count = 0
        self.in_rep = False
        self.feedback_messages = []


class PushUpAnalyzer(ExerciseAnalyzer):
    """
    Analyzes push-ups using elbow angle and form validation.
    
    Rep detection logic:
    - DOWN phase: Elbow angle decreases to < 100°
    - UP phase: Elbow angle increases to > 150°
    - One rep counted when transitioning from UP -> DOWN -> UP
    """
    
    # Angle thresholds for rep detection
    ELBOW_DOWN_THRESHOLD = 100  # Degrees - push-up at bottom
    ELBOW_UP_THRESHOLD = 150    # Degrees - push-up at top
    
    # Form validation thresholds
    MIN_DEPTH_ANGLE = 90         # Elbow must go below this for valid rep
    BACK_ANGLE_TOLERANCE = 30    # Degrees - how straight back should be
    
    def __init__(self, pose_detector: PoseDetector):
        """Initialize push-up analyzer."""
        super().__init__(pose_detector)
        self.last_elbow_angle = None
        self.rep_phase = "up"  # Track phase: "up" or "down"
        self.max_depth_angle = 180  # Track minimum angle in current rep
    
    def get_required_landmarks(self) -> List[str]:
        """Push-ups require shoulder, elbow, wrist, hip, knee visibility."""
        return [
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE"
        ]
    
    def analyze(self, landmarks: np.ndarray, visibility: np.ndarray) -> Dict:
        """
        Analyze push-up form and count reps.
        
        Args:
            landmarks: Nx3 array of pose landmarks
            visibility: N array of visibility scores
            
        Returns:
            Analysis results
        """
        self.feedback_messages = []
        
        # Validate required landmarks are visible
        if not self.pose_detector.is_pose_valid(visibility, self.get_required_landmarks(), 0.5):
            return {
                'rep_count': self.rep_count,
                'in_rep': False,
                'feedback': ["Ensure shoulders, elbows, and hips are visible"],
                'form_status': {}
            }
        
        # Calculate key angles
        left_elbow_angle = self._calculate_elbow_angle(landmarks, "left")
        right_elbow_angle = self._calculate_elbow_angle(landmarks, "right")
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Track depth (minimum angle in rep)
        if avg_elbow_angle < self.max_depth_angle:
            self.max_depth_angle = avg_elbow_angle
        
        # Rep detection state machine
        self._detect_rep(avg_elbow_angle)
        
        # Form validation
        form_issues = self._validate_form(landmarks, visibility, avg_elbow_angle)
        
        return {
            'rep_count': self.rep_count,
            'in_rep': self.in_rep,
            'feedback': self.feedback_messages,
            'form_status': {
                'avg_elbow_angle': avg_elbow_angle,
                'left_elbow_angle': left_elbow_angle,
                'right_elbow_angle': right_elbow_angle,
                'form_issues': form_issues
            }
        }
    
    def _calculate_elbow_angle(self, landmarks: np.ndarray, side: str) -> float:
        """
        Calculate elbow angle for a given side.
        
        Args:
            landmarks: Pose landmarks
            side: "left" or "right"
            
        Returns:
            Elbow angle in degrees
        """
        if side == "left":
            shoulder = self.pose_detector.get_landmark(landmarks, "LEFT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "LEFT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "LEFT_WRIST")
        else:
            shoulder = self.pose_detector.get_landmark(landmarks, "RIGHT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "RIGHT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "RIGHT_WRIST")
        
        angle = self.pose_detector.calculate_angle(shoulder, elbow, wrist)
        return angle
    
    def _detect_rep(self, avg_elbow_angle: float):
        """
        Detect and count reps using state machine.
        
        Args:
            avg_elbow_angle: Average elbow angle of both arms
        """
        # State machine:
        # UP (angle > 150) -> DOWN (angle < 100) -> UP = 1 rep counted
        
        if self.rep_phase == "up" and avg_elbow_angle < self.ELBOW_DOWN_THRESHOLD:
            # Transitioned to DOWN phase
            self.rep_phase = "down"
            self.in_rep = True
            self.max_depth_angle = avg_elbow_angle
            self.feedback_messages.append("Going down...")
        
        elif self.rep_phase == "down" and avg_elbow_angle > self.ELBOW_UP_THRESHOLD:
            # Transitioned back to UP phase - rep complete!
            self.rep_phase = "up"
            self.rep_count += 1
            self.in_rep = False
            self.max_depth_angle = 180
            self.feedback_messages.append(f"Rep {self.rep_count} complete!")
    
    def _validate_form(self, landmarks: np.ndarray, visibility: np.ndarray, 
                      avg_elbow_angle: float) -> List[str]:
        """
        Validate push-up form and return feedback.
        
        Args:
            landmarks: Pose landmarks
            visibility: Visibility scores
            avg_elbow_angle: Current average elbow angle
            
        Returns:
            List of form issues
        """
        issues = []
        
        # Check depth (elbow must go below 90 degrees)
        if self.in_rep and avg_elbow_angle > self.MIN_DEPTH_ANGLE:
            issues.append("Go deeper! Elbows should bend more")
        
        # Check back alignment
        back_angle = self._calculate_back_angle(landmarks)
        if abs(back_angle) > self.BACK_ANGLE_TOLERANCE:
            if back_angle > 0:
                issues.append("Back is sagging - keep it straight!")
            else:
                issues.append("Back is arching - keep it neutral!")
        
        # Check arm symmetry
        left_elbow = self._calculate_elbow_angle(landmarks, "left")
        right_elbow = self._calculate_elbow_angle(landmarks, "right")
        if abs(left_elbow - right_elbow) > 15:
            issues.append("Arms are uneven - keep balanced")
        
        self.feedback_messages.extend(issues)
        return issues
    
    def _calculate_back_angle(self, landmarks: np.ndarray) -> float:
        """
        Calculate back angle (spine alignment).
        Positive = sagging, Negative = arching, ~0 = neutral.
        
        Args:
            landmarks: Pose landmarks
            
        Returns:
            Back angle deviation in degrees
        """
        shoulder = (
            self.pose_detector.get_landmark(landmarks, "LEFT_SHOULDER") +
            self.pose_detector.get_landmark(landmarks, "RIGHT_SHOULDER")
        ) / 2
        hip = (
            self.pose_detector.get_landmark(landmarks, "LEFT_HIP") +
            self.pose_detector.get_landmark(landmarks, "RIGHT_HIP")
        ) / 2
        
        # Calculate spine vector (should be vertical when doing push-ups)
        spine = hip - shoulder
        
        # Ideal back is vertical (pointing down in image)
        # Calculate deviation from vertical using x-component
        # If spine[0] is large, back is leaning
        back_angle = np.degrees(np.arctan2(spine[0], -spine[1]))
        
        return back_angle


class BicepCurlAnalyzer(ExerciseAnalyzer):
    """
    Analyzes bicep curls using elbow flexion angle.
    
    Rep detection logic:
    - DOWN phase: Elbow angle increases to > 160° (arm extended)
    - UP phase: Elbow angle decreases to < 60° (arm flexed)
    - One rep counted when transitioning from DOWN -> UP -> DOWN
    """
    
    ELBOW_DOWN_THRESHOLD = 160  # Extended arm
    ELBOW_UP_THRESHOLD = 60     # Flexed arm
    
    def __init__(self, pose_detector: PoseDetector):
        """Initialize bicep curl analyzer."""
        super().__init__(pose_detector)
        self.rep_phase = "down"  # Start in down phase
    
    def get_required_landmarks(self) -> List[str]:
        """Bicep curls require shoulder, elbow, wrist visibility."""
        return [
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST"
        ]
    
    def analyze(self, landmarks: np.ndarray, visibility: np.ndarray) -> Dict:
        """
        Analyze bicep curl form and count reps.
        
        Args:
            landmarks: Nx3 array of pose landmarks
            visibility: N array of visibility scores
            
        Returns:
            Analysis results
        """
        self.feedback_messages = []
        
        # Validate required landmarks
        if not self.pose_detector.is_pose_valid(visibility, self.get_required_landmarks(), 0.5):
            return {
                'rep_count': self.rep_count,
                'in_rep': False,
                'feedback': ["Ensure arms are visible"],
                'form_status': {}
            }
        
        # Calculate average elbow angle for both arms
        left_elbow = self._calculate_elbow_angle(landmarks, "left")
        right_elbow = self._calculate_elbow_angle(landmarks, "right")
        avg_elbow_angle = (left_elbow + right_elbow) / 2
        
        # Rep detection
        self._detect_rep(avg_elbow_angle)
        
        # Form validation
        form_issues = self._validate_form(left_elbow, right_elbow)
        
        return {
            'rep_count': self.rep_count,
            'in_rep': self.in_rep,
            'feedback': self.feedback_messages,
            'form_status': {
                'avg_elbow_angle': avg_elbow_angle,
                'left_elbow_angle': left_elbow,
                'right_elbow_angle': right_elbow,
                'form_issues': form_issues
            }
        }
    
    def _calculate_elbow_angle(self, landmarks: np.ndarray, side: str) -> float:
        """
        Calculate elbow angle for a given side.
        
        Args:
            landmarks: Pose landmarks
            side: "left" or "right"
            
        Returns:
            Elbow angle in degrees
        """
        if side == "left":
            shoulder = self.pose_detector.get_landmark(landmarks, "LEFT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "LEFT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "LEFT_WRIST")
        else:
            shoulder = self.pose_detector.get_landmark(landmarks, "RIGHT_SHOULDER")
            elbow = self.pose_detector.get_landmark(landmarks, "RIGHT_ELBOW")
            wrist = self.pose_detector.get_landmark(landmarks, "RIGHT_WRIST")
        
        angle = self.pose_detector.calculate_angle(shoulder, elbow, wrist)
        return angle
    
    def _detect_rep(self, avg_elbow_angle: float):
        """
        Detect and count reps using state machine.
        
        Args:
            avg_elbow_angle: Average elbow angle of both arms
        """
        if self.rep_phase == "down" and avg_elbow_angle < self.ELBOW_UP_THRESHOLD:
            # Started curling up
            self.rep_phase = "up"
            self.in_rep = True
            self.feedback_messages.append("Curling up...")
        
        elif self.rep_phase == "up" and avg_elbow_angle > self.ELBOW_DOWN_THRESHOLD:
            # Returned to down - rep complete!
            self.rep_phase = "down"
            self.rep_count += 1
            self.in_rep = False
            self.feedback_messages.append(f"Rep {self.rep_count} complete!")
    
    def _validate_form(self, left_elbow: float, right_elbow: float) -> List[str]:
        """
        Validate bicep curl form.
        
        Args:
            left_elbow: Left arm elbow angle
            right_elbow: Right arm elbow angle
            
        Returns:
            List of form issues
        """
        issues = []
        
        # Check arm symmetry
        if abs(left_elbow - right_elbow) > 15:
            issues.append("Arms are uneven - keep balanced")
        
        self.feedback_messages.extend(issues)
        return issues
