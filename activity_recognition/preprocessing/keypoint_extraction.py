import cv2
import mediapipe as mp
import numpy as np
from .angle_calculation import calculate_angle
  # Liste explicite des exports
ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
]
IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW",
    "LEFT_ELBOW", "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"
]
angle_headers = [f"{a}_{b}_{c}_angle" for a, b, c in ANGLE_JOINTS]
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
HEADERS = ["label"]
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
# Création des colonnes du CSV
# 3. Fonction : Redimensionner l'image
HEADERS = HEADERS+ angle_headers 
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
# Fonction pour extraire les keypoints
def extract_keypoints(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in IMPORTANT_LMS:
            point = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.append([point.x, point.y, point.z, point.visibility])

        normalized_keypoints = normalize_keypoints(keypoints)
        normalized_keypoints = np.array(normalized_keypoints).flatten()

        # Calculate angles and append them only once
        angles = []
        for joint1, joint2, joint3 in ANGLE_JOINTS:
            a = landmarks[mp_pose.PoseLandmark[joint1].value]
            b = landmarks[mp_pose.PoseLandmark[joint2].value]
            c = landmarks[mp_pose.PoseLandmark[joint3].value]

            angle = calculate_angle([a.x, a.y], [b.x, b.y], [c.x, c.y])
            angles.append(angle)

        # Concatenate keypoints and angles
        all_features = np.concatenate([normalized_keypoints, angles])

        return all_features
