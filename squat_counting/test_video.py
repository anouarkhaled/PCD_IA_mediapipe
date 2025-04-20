import cv2
import mediapipe as mp
import numpy as np
import joblib 
model2=joblib.load( 'model_filename.pkl')
label_encoder=joblib.load('label_encoder.pkl')
# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
Landmark = mp.solutions.pose.PoseLandmark
mp_drawing = mp.solutions.drawing_utils

# Fonction pour calculer l’angle entre trois points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Fonction d’extraction des angles
def extract_angles_from_landmarks(landmarks):
    angles = {}

    angles['left_knee'] = calculate_angle(
        [landmarks[Landmark.LEFT_HIP].x, landmarks[Landmark.LEFT_HIP].y],
        [landmarks[Landmark.LEFT_KNEE].x, landmarks[Landmark.LEFT_KNEE].y],
        [landmarks[Landmark.LEFT_ANKLE].x, landmarks[Landmark.LEFT_ANKLE].y]
    )

    angles['left_hip'] = calculate_angle(
        [landmarks[Landmark.LEFT_SHOULDER].x, landmarks[Landmark.LEFT_SHOULDER].y],
        [landmarks[Landmark.LEFT_HIP].x, landmarks[Landmark.LEFT_HIP].y],
        [landmarks[Landmark.LEFT_KNEE].x, landmarks[Landmark.LEFT_KNEE].y]
    )

    angles['right_hip'] = calculate_angle(
        [landmarks[Landmark.RIGHT_SHOULDER].x, landmarks[Landmark.RIGHT_SHOULDER].y],
        [landmarks[Landmark.RIGHT_HIP].x, landmarks[Landmark.RIGHT_HIP].y],
        [landmarks[Landmark.RIGHT_KNEE].x, landmarks[Landmark.RIGHT_KNEE].y]
    )

    angles['right_knee'] = calculate_angle(
        [landmarks[Landmark.RIGHT_HIP].x, landmarks[Landmark.RIGHT_HIP].y],
        [landmarks[Landmark.RIGHT_KNEE].x, landmarks[Landmark.RIGHT_KNEE].y],
        [landmarks[Landmark.RIGHT_ANKLE].x, landmarks[Landmark.RIGHT_ANKLE].y]
    )

    return angles
def test_webcam():
    cap = cv2.VideoCapture(0)  # 0 = caméra par défaut

    if not cap.isOpened():
        print("❌ Impossible d’accéder à la webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Erreur de lecture de la webcam.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = extract_angles_from_landmarks(landmarks)

            # Préparer les features pour prédiction
            features = np.array([[angles['left_knee'], angles['left_hip'], angles['right_hip'], angles['right_knee']]])
            prediction = model2.predict(features)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Affichage
            cv2.putText(frame, f'Phase: {predicted_label}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Webcam - Prédiction en temps réel", frame)

        # Quitter avec 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Analyse de la vidéo
def test_video(video_path):
    new_prediction=""
    old_prediction=""
    compteur=0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Impossible d’ouvrir la vidéo : {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        old_prediction=new_prediction
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = extract_angles_from_landmarks(landmarks)

            # Préparation des données pour la prédiction
            features = np.array([[angles['left_knee'], angles['left_hip'], angles['right_hip'], angles['right_knee']]])
            prediction = model2.predict(features)
            
            new_prediction = label_encoder.inverse_transform(prediction)[0]
            if (new_prediction=="haut"and old_prediction=="milieu"):
                compteur+=1
            
            print(new_prediction)
            print("Angles:", features)


            # Affichage du label sur la vidéo
            cv2.putText(frame, f'Phase: {new_prediction} compteur={compteur}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Dessiner les articulations
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Phase Squat - Prédiction", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Lancer l’analyse
if __name__ == "__main__":
    test_video("squat_counting/squat_25.mp4")
