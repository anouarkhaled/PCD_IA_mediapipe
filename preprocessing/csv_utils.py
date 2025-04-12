import os
import csv
import numpy as np
from preprocessing.keypoint_extraction import HEADERS
from config import IMPORTANT_LMS
import mediapipe as mp
mp_pose = mp.solutions.pose
def init_csv(dataset_path: str):
    if os.path.exists(dataset_path):
        return  # Ne rien faire si le fichier existe déjà
    with open(dataset_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(HEADERS)

def export_keypoints_to_csv(path, keypoints, label):
    keypoints.insert(0, label)
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keypoints)
def export_landmark_to_csv(dataset_path: str, results, label: str):
    try:
        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in IMPORTANT_LMS:
            point = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.append([point.x, point.y, point.z, point.visibility])

        keypoints = list(np.array(keypoints).flatten())
        keypoints.insert(0, label)  # Ajouter le label au début

        with open(dataset_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(keypoints)

    except Exception as e:
        print("Erreur:", e)
