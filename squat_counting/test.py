# Faire des prédictions sur l'ensemble de test
from model_training import model
from model_training import X_test, classification_report,y_test,label_encoder

y_pred = model.predict(X_test)

# Évaluer les performances du modèle
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
