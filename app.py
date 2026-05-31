

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

print("Loading dataset and training models...")
try:
  
    data = pd.read_csv("diabetes.csv")
    
    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_resample(x, y)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)
    
    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM":               SVC(probability=True, kernel='rbf', C=1, random_state=42),
    }
    
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        score = model.score(x_test_scaled, y_test)
        print(f"  ✓ {name}: {score:.3f} accuracy")
    
    FEATURE_COLUMNS = list(x.columns)  
    
    print(f"✓ Models ready | Features: {FEATURE_COLUMNS}")
    MODELS_LOADED = True

except Exception as e:
    print(f" Could not load dataset: {e}")
    print("  → Running in DEMO mode (random predictions)")
    MODELS_LOADED = False
    FEATURE_COLUMNS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    scaler = None
    models = {}



def apply_krr_rules(patient):
    """Pure-Python version of your SWRL rules."""
    risk_factors = []
    conditions   = []
    inferred     = []

    glucose = patient.get('glucose', 0)
    bmi     = patient.get('bmi', 0)
    age     = patient.get('age', 0)
    bp      = patient.get('blood_pressure', 0)

   
    if glucose > 180:
        conditions.append("High Blood Sugar")
        inferred.append("HighBloodSugar")

  
    if bmi > 30:
        risk_factors.append("Obesity")

   
    elif 25 < bmi <= 30:
        risk_factors.append("Overweight / Physical Inactivity")

  
    if 140 < glucose <= 180 and age > 35:
        inferred.append("Pre-Diabetic")
        conditions.append("Pre-Diabetic Condition")

    if bp > 140:
        risk_factors.append("Hypertension")

    if len(risk_factors) >= 2:
        inferred.append("HighRiskPatient")

    if "HighRiskPatient" in inferred and glucose > 200:
        conditions.append("Likely Type 2 Diabetes")

    return {
        "risk_factors": risk_factors,
        "conditions":   conditions,
        "inferred_classes": inferred,
        "is_high_risk": "HighRiskPatient" in inferred,
    }


def generate_recommendations(krr, hybrid_score, ml_pred):
    recs = []
    if hybrid_score > 0.70:
        recs.append({"level": "danger",  "text": "HIGH RISK — Schedule an immediate clinical consultation"})
        recs.append({"level": "danger",  "text": "Required: HbA1c test and fasting glucose retest"})
    elif hybrid_score > 0.40:
        recs.append({"level": "warning", "text": "MODERATE RISK — Follow-up appointment in 3–6 months recommended"})
        recs.append({"level": "warning", "text": "Consider lifestyle modifications: diet and exercise"})
    else:
        recs.append({"level": "success", "text": "LOW RISK — Routine annual screening is sufficient"})

    if "Obesity" in krr["risk_factors"]:
        recs.append({"level": "warning", "text": "Weight management program recommended (BMI > 30)"})
    if "Hypertension" in krr["risk_factors"]:
        recs.append({"level": "warning", "text": "Monitor blood pressure regularly"})
    if "High Blood Sugar" in krr["conditions"]:
        recs.append({"level": "danger",  "text": "Monitor glucose levels closely — levels above 180 mg/dL detected"})
    if "Pre-Diabetic Condition" in krr["conditions"]:
        recs.append({"level": "warning", "text": "Pre-diabetic indicators detected — dietary changes advised"})

    return recs



@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models_loaded": MODELS_LOADED, "features": FEATURE_COLUMNS})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "No JSON body"}), 400

        patient = {
            "pregnancies":  float(body.get("pregnancies", 0)),
            "glucose":      float(body.get("glucose", 0)),
            "blood_pressure": float(body.get("blood_pressure", 0)),
            "skin_thickness": float(body.get("skin_thickness", 0)),
            "insulin":      float(body.get("insulin", 0)),
            "bmi":          float(body.get("bmi", 0)),
            "dpf":          float(body.get("dpf", 0)),
            "age":          float(body.get("age", 0)),
        }

        krr = apply_krr_rules(patient)

        ml_results = {}
        if MODELS_LOADED:
            feature_vector = np.array([[
                patient["pregnancies"],
                patient["glucose"],
                patient["blood_pressure"],
                patient["skin_thickness"],
                patient["insulin"],
                patient["bmi"],
                patient["dpf"],
                patient["age"],
            ]])
            feature_scaled = scaler.transform(feature_vector)

            for name, model in models.items():
                proba = model.predict_proba(feature_scaled)[0]
                pred  = int(model.predict(feature_scaled)[0])
                ml_results[name] = {
                    "prediction": pred,
                    "probability_diabetic":     round(float(proba[1]) * 100, 1),
                    "probability_non_diabetic": round(float(proba[0]) * 100, 1),
                }
        else:
            import random
            for name in ["Random Forest", "Logistic Regression", "Gradient Boosting", "SVM"]:
                p = round(random.uniform(20, 80), 1)
                ml_results[name] = {
                    "prediction": 1 if p > 50 else 0,
                    "probability_diabetic": p,
                    "probability_non_diabetic": round(100 - p, 1),
                }

        best = ml_results.get("Random Forest", list(ml_results.values())[0])
        ml_prob   = best["probability_diabetic"] / 100.0
        ml_pred   = best["prediction"]

        krr_risk = len(krr["risk_factors"]) + len(krr["conditions"])
        hybrid_score = round(0.4 * min(krr_risk / 5, 1.0) + 0.6 * ml_prob, 4)

        if hybrid_score > 0.70:
            risk_level = "High"
        elif hybrid_score > 0.40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        recs = generate_recommendations(krr, hybrid_score, ml_pred)

        return jsonify({
            "patient": patient,
            "krr": krr,
            "ml_models": ml_results,
            "hybrid_score": round(hybrid_score * 100, 1),
            "risk_level": risk_level,
            "ml_prediction": ml_pred,
            "recommendations": recs,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n🏥 Diabetes Detection API starting...")
    print("   Open index.html in your browser to use the UI\n")
    app.run(debug=True, port=5000)
