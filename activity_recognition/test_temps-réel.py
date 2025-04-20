import cv2
import numpy as np
import mediapipe as mp
from model.predict import predict_activity  # ta fonction de prédiction
from preprocessing.extract import extract_keypoints  # ta fonction d'extraction

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Paramètres
SEQUENCE_LENGTH = 20
sequence = []

# Capture webcam (ou vidéo)
cap = cv2.VideoCapture(0)  # mettre le chemin de la vidéo si ce n'est pas la webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extraire les keypoints de l'image actuelle
    keypoints = extract_keypoints(frame)

    if keypoints is not None:
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:
            sequence_np = np.array(sequence)
            activity = predict_activity(sequence_np)  # ton modèle prédit ici
            sequence.pop(0)  # on garde la taille de la séquence constante

            # Afficher le texte sur l’image
            cv2.putText(frame, f'Activité : {activity}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        # Si pas de détection : on affiche que rien n’est détecté
        cv2.putText(frame, 'Aucune pose détectée', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    # Affichage de la vidéo en temps réel
    cv2.imshow('Reconnaissance d\'activités sportives', frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
