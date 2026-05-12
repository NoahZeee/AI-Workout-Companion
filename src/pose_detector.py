"""
pose_detector.py - Real Pose Detection using MediaPipe

Uses MediaPipe Pose Landmarker for accurate real-time pose estimation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import os
import urllib.request


class PoseDetector:
    """
    Real-time pose detector using MediaPipe Pose Landmarker.
    
    Outputs 17 keypoints in COCO format:
    0: Nose, 1-2: Eyes, 3-4: Ears, 5-6: Shoulders, 7-8: Elbows, 9-10: Wrists,
    11-12: Hips, 13-14: Knees, 15-16: Ankles
    """
    
    # COCO 17-point landmark indices
    LANDMARKS = {
        "NOSE": 0,
        "LEFT_EYE": 1,
        "RIGHT_EYE": 2,
        "LEFT_EAR": 3,
        "RIGHT_EAR": 4,
        "LEFT_SHOULDER": 5,
        "RIGHT_SHOULDER": 6,
        "LEFT_ELBOW": 7,
        "RIGHT_ELBOW": 8,
        "LEFT_WRIST": 9,
        "RIGHT_WRIST": 10,
        "LEFT_HIP": 11,
        "RIGHT_HIP": 12,
        "LEFT_KNEE": 13,
        "RIGHT_KNEE": 14,
        "LEFT_ANKLE": 15,
        "RIGHT_ANKLE": 16,
    }
    
    # MediaPipe 33-point to COCO 17-point mapping
    # MediaPipe indices: https://mediapipe.dev/images/mobile/pose_tracking_full_body_landmarks.png
    MEDIAPIPE_TO_COCO = [
        0,   # 0: Nose -> 0
        2,   # 1: Left Eye -> 1
        5,   # 2: Right Eye -> 2
        7,   # 3: Left Ear -> 3
        8,   # 4: Right Ear -> 4
        11,  # 5: Left Shoulder -> 5
        12,  # 6: Right Shoulder -> 6
        13,  # 7: Left Elbow -> 7
        14,  # 8: Right Elbow -> 8
        15,  # 9: Left Wrist -> 9
        16,  # 10: Right Wrist -> 10
        23,  # 11: Left Hip -> 11
        24,  # 12: Right Hip -> 12
        25,  # 13: Left Knee -> 13
        26,  # 14: Right Knee -> 14
        27,  # 15: Left Ankle -> 15
        28,  # 16: Right Ankle -> 16
    ]
    
    def __init__(self, min_detection_confidence: float = 0.7, 
                 min_tracking_confidence: float = 0.7):
        """
        Initialize pose detector using MediaPipe.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection [0-1]
            min_tracking_confidence: Minimum confidence for tracking [0-1]
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose_landmarker = None
        self.frame_count = 0
        
        self._initialize_model()
        print("[OK] Real pose detector initialized (MediaPipe Pose Landmarker)")
    
    def _initialize_model(self):
        """Initialize MediaPipe Pose Landmarker model."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            print("  [*] Loading MediaPipe Pose Landmarker model...")
            
            # Model path
            model_path = 'data/pose_landmarker_lite.task'
            
            # Download if missing
            if not os.path.exists(model_path):
                print("      Model not found. Downloading (~7 MB)...")
                self._download_model(model_path)
            
            # Create pose landmarker
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                num_poses=1
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            print("  [OK] MediaPipe Pose Landmarker loaded successfully!")
            
        except ImportError as e:
            print(f"  [!] Error: MediaPipe not installed: {e}")
            print("      Please run: pip install mediapipe opencv-python numpy")
            raise
        except Exception as e:
            print(f"  [!] Error initializing MediaPipe: {e}")
            raise
    
    def _download_model(self, model_path: str):
        """Download MediaPipe pose landmarker model."""
        try:
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            
            # Create data directory if needed
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 // total_size, 100)
                    print(f"\r      [{percent:3d}%] {downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB", end='')
            
            urllib.request.urlretrieve(url, model_path, download_progress)
            print(f"\n      [OK] Model downloaded to {model_path}")
            
        except Exception as e:
            print(f"  [!] Download failed: {e}")
            raise
    
    def detect_pose(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pose landmarks in a frame.
        
        Args:
            frame: BGR image frame (OpenCV format)
            
        Returns:
            Tuple of (landmarks, visibility):
            - landmarks: Nx3 array of [x, y, z] in normalized coordinates [0, 1]
            - visibility: N array of confidence scores [0, 1]
            Returns (None, None) if no pose detected
        """
        try:
            import mediapipe as mp
            
            self.frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Run detection
            detection_result = self.pose_landmarker.detect(mp_image)
            
            # Check if pose detected
            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None, None
            
            # Get first detected pose
            mediapipe_landmarks = detection_result.pose_landmarks[0]
            
            # Map from MediaPipe 33-point to COCO 17-point
            landmarks = []
            visibility = []
            
            for coco_idx in self.MEDIAPIPE_TO_COCO:
                lm = mediapipe_landmarks[coco_idx]
                landmarks.append([lm.x, lm.y, lm.z])
                visibility.append(lm.visibility)
            
            landmarks = np.array(landmarks, dtype=np.float32)
            visibility = np.array(visibility, dtype=np.float32)
            
            # Validate landmarks are in valid range
            if np.any(np.isnan(landmarks)) or np.any(np.isnan(visibility)):
                return None, None
            
            return landmarks, visibility
            
        except Exception as e:
            print(f"[!] Error in pose detection: {e}")
            return None, None
    
    def calculate_angle(self, point_a: np.ndarray, point_b: np.ndarray, 
                       point_c: np.ndarray) -> float:
        """
        Calculate angle at point_b formed by points a-b-c.
        
        Args:
            point_a: First point [x, y, z]
            point_b: Vertex point [x, y, z]
            point_c: Third point [x, y, z]
            
        Returns:
            Angle in degrees (0-180)
        """
        # Vectors from b to a and b to c (use only x, y)
        ba = point_a[:2] - point_b[:2]
        bc = point_c[:2] - point_b[:2]
        
        # Compute angle
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def get_landmark(self, landmarks: np.ndarray, landmark_name: str) -> np.ndarray:
        """
        Get a specific landmark by name.
        
        Args:
            landmarks: Nx3 array of pose landmarks
            landmark_name: Name of landmark (see LANDMARKS dict)
            
        Returns:
            3D coordinates of the landmark
        """
        idx = self.LANDMARKS.get(landmark_name)
        if idx is None:
            raise ValueError(f"Unknown landmark: {landmark_name}")
        return landmarks[idx]
    
    def get_visibility(self, visibility: np.ndarray, landmark_name: str) -> float:
        """
        Get visibility/confidence for a specific landmark.
        
        Args:
            visibility: N array of confidence scores
            landmark_name: Name of landmark
            
        Returns:
            Confidence score [0-1]
        """
        idx = self.LANDMARKS.get(landmark_name)
        if idx is None:
            raise ValueError(f"Unknown landmark: {landmark_name}")
        return visibility[idx]
    
    def draw_pose(self, frame: np.ndarray, landmarks: np.ndarray, 
                  visibility: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Draw pose skeleton on the frame.
        
        Args:
            frame: BGR image frame
            landmarks: Nx3 array of normalized [x, y, z] coordinates
            visibility: N array of confidence scores
            confidence_threshold: Only draw keypoints with visibility >= threshold
            
        Returns:
            Frame with pose skeleton drawn
        """
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        frame_landmarks = []
        for lm, vis in zip(landmarks, visibility):
            if vis >= confidence_threshold:
                x = int(lm[0] * w)
                y = int(lm[1] * h)
                # Clamp to frame bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                frame_landmarks.append((x, y, float(vis)))
            else:
                frame_landmarks.append(None)
        
        # Pose skeleton connections (COCO 17-point format)
        connections = [
            (0, 1), (0, 2),           # Nose to eyes
            (1, 3), (2, 4),           # Eyes to ears
            (5, 6),                   # Shoulders
            (5, 7), (7, 9),           # Left arm
            (6, 8), (8, 10),          # Right arm
            (5, 11), (6, 12),         # Shoulders to hips
            (11, 12),                 # Hips
            (11, 13), (13, 15),       # Left leg
            (12, 14), (14, 16),       # Right leg
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if frame_landmarks[start_idx] is not None and frame_landmarks[end_idx] is not None:
                x1, y1, _ = frame_landmarks[start_idx]
                x2, y2, _ = frame_landmarks[end_idx]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw keypoints
        for idx, landmark in enumerate(frame_landmarks):
            if landmark is not None:
                x, y, vis = landmark
                # Color intensity based on visibility
                color_val = int(200 * vis)
                cv2.circle(frame, (x, y), 5, (0, color_val, 255 - color_val), -1)
        
        return frame
    
    def is_pose_valid(self, visibility: np.ndarray, required_landmarks: List[str], 
                     confidence_threshold: float = 0.5) -> bool:
        """
        Check if pose is valid by verifying required landmarks have sufficient visibility.
        
        Args:
            visibility: N array of confidence scores
            required_landmarks: List of landmark names that must be visible
            confidence_threshold: Minimum visibility threshold
            
        Returns:
            True if all required landmarks are visible
        """
        for landmark_name in required_landmarks:
            if self.get_visibility(visibility, landmark_name) < confidence_threshold:
                return False
        return True    
    def detect_visible_side(self, visibility: np.ndarray) -> str:
        """
        Detect which side of the body is more visible (left or right).
        Used for side-view exercises where only one side is clearly visible.
        
        Args:
            visibility: N array of confidence scores
            
        Returns:
            'left', 'right', or 'both' based on visibility of arm landmarks
        """
        try:
            left_elbow_vis = self.get_visibility(visibility, "LEFT_ELBOW")
            right_elbow_vis = self.get_visibility(visibility, "RIGHT_ELBOW")
            left_wrist_vis = self.get_visibility(visibility, "LEFT_WRIST")
            right_wrist_vis = self.get_visibility(visibility, "RIGHT_WRIST")
            
            left_arm_vis = (left_elbow_vis + left_wrist_vis) / 2
            right_arm_vis = (right_elbow_vis + right_wrist_vis) / 2
            
            # Threshold: consider an arm visible if avg visibility > 0.2
            left_visible = left_arm_vis > 0.2
            right_visible = right_arm_vis > 0.2
            
            if left_visible and not right_visible:
                return 'left'
            elif right_visible and not left_visible:
                return 'right'
            else:
                return 'both'
        except:
            return 'both'