import cv2
import mediapipe as mp
import numpy as np
  # Liste explicite des exports
IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW",
    "LEFT_ELBOW", "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"
]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
HEADERS = ["label"]
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
# Création des colonnes du CSV
# 3. Fonction : Redimensionner l'image
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
# Fonction pour extraire les keypoints
def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for lm_name in IMPORTANT_LMS:
            lm = landmarks[mp_pose.PoseLandmark[lm_name].value]
            keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints