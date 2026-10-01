# GlucoSense - AI-Based Diabetes Risk Detection & Clinical Decision Support System

GlucoSense is an AI-based diabetes risk assessment system that combines multiple machine learning classification models with a knowledge/rule-based reasoning layer to estimate diabetes risk from patient health indicators.

The system provides a Flask REST API that accepts patient information, runs multiple machine learning models, applies predefined rules, and produces a combined hybrid risk score, risk level, inferred conditions, and recommendations.

IMPORTANT: GlucoSense is an educational/research project and is NOT a medical diagnostic tool. Predictions and recommendations should not be used as a substitute for professional medical advice.

--------------------------------------------------
PROJECT OVERVIEW
--------------------------------------------------

Diabetes risk assessment can involve multiple health indicators such as glucose level, BMI, blood pressure, age, and other patient characteristics.

GlucoSense uses two complementary approaches:

1. MACHINE LEARNING
   - Random Forest
   - Logistic Regression
   - Gradient Boosting
   - Support Vector Machine (SVM)

2. KNOWLEDGE-BASED REASONING
   - High blood sugar
   - Obesity
   - Overweight
   - Hypertension
   - Pre-diabetic indicators
   - High-risk patient status
   - Likely Type 2 Diabetes

The outputs from both approaches are combined into a hybrid risk score.

--------------------------------------------------
SYSTEM ARCHITECTURE
--------------------------------------------------

Patient Input
     |
     v
Flask API (/api/predict)
     |
     +-------------------------+
     |                         |
     v                         v
Machine Learning          Rule-Based / KRR
Models                    Reasoning
     |                         |
     |                         |
     +------------+------------+
                  |
                  v
          Hybrid Risk Score
                  |
                  v
            Risk Level
         Low / Moderate / High
                  |
                  v
       Recommendations &
       Inferred Conditions

--------------------------------------------------
DATASET
--------------------------------------------------

The project uses the Pima Indians Diabetes Dataset, a commonly used dataset for binary diabetes classification.

Input features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:

- Outcome

Target encoding:

0 = Non-Diabetic
1 = Diabetic

--------------------------------------------------
MACHINE LEARNING PIPELINE
--------------------------------------------------

The backend performs the following steps:

1. Load diabetes.csv
2. Separate features and target
3. Apply SMOTE
4. Split data into training and testing sets
5. Standardize features using StandardScaler
6. Train four machine learning models
7. Evaluate models
8. Use the trained models for prediction

SMOTE is used to address class imbalance:

    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_resample(x, y)

Feature scaling is performed with StandardScaler:

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

--------------------------------------------------
MACHINE LEARNING MODELS
--------------------------------------------------

1. RANDOM FOREST

    RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )

2. LOGISTIC REGRESSION

    LogisticRegression(
        max_iter=1000,
        random_state=42
    )

3. GRADIENT BOOSTING

    GradientBoostingClassifier(
        n_estimators=200,
        random_state=42
    )

4. SUPPORT VECTOR MACHINE

    SVC(
        probability=True,
        kernel='rbf',
        C=1,
        random_state=42
    )

Each model provides:

- Binary prediction
- Probability of diabetes
- Probability of non-diabetes

--------------------------------------------------
KNOWLEDGE-BASED REASONING
--------------------------------------------------

The apply_krr_rules() function evaluates patient information using predefined rules inspired by Knowledge Representation and Reasoning (KRR).

Example rules:

HIGH BLOOD SUGAR

    IF glucose > 180
    THEN High Blood Sugar

OBESITY

    IF BMI > 30
    THEN Obesity

OVERWEIGHT

    IF 25 < BMI <= 30
    THEN Overweight / Physical Inactivity

PRE-DIABETIC INDICATORS

    IF 140 < glucose <= 180
    AND age > 35
    THEN Pre-Diabetic

HYPERTENSION

    IF blood pressure > 140
    THEN Hypertension

HIGH-RISK PATIENT

    IF at least two risk factors/conditions are detected
    THEN HighRiskPatient

LIKELY TYPE 2 DIABETES

    IF HighRiskPatient
    AND glucose > 200
    THEN Likely Type 2 Diabetes

--------------------------------------------------
HYBRID RISK ASSESSMENT
--------------------------------------------------

The system combines the machine learning prediction with rule-based reasoning.

The Random Forest probability is currently used as the primary ML probability:

    ml_prob = best["probability_diabetic"] / 100.0

The rule-based risk contribution is calculated from the number of detected risk factors and conditions.

The hybrid score is calculated as:

    hybrid_score =
        0.4 * min(krr_risk / 5, 1.0)
        + 0.6 * ml_prob

Therefore:

    60% = Machine Learning probability
    40% = Rule-based reasoning

Risk levels:

    Score <= 40%       = Low
    Score > 40%-70%    = Moderate
    Score > 70%        = High

--------------------------------------------------
FLASK REST API
--------------------------------------------------

The backend is implemented using Flask.

CORS is enabled so that a separate frontend can communicate with the API.

The server runs on:

    http://localhost:5000

--------------------------------------------------
HEALTH ENDPOINT
--------------------------------------------------

GET /api/health

Returns the API and model status.

Example:

    {
      "status": "ok",
      "models_loaded": true,
      "features": [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
      ]
    }

--------------------------------------------------
PREDICTION ENDPOINT
--------------------------------------------------

POST /api/predict

The endpoint accepts patient information as JSON.

Example request:

    {
      "pregnancies": 2,
      "glucose": 145,
      "blood_pressure": 85,
      "skin_thickness": 25,
      "insulin": 100,
      "bmi": 31.5,
      "dpf": 0.5,
      "age": 42
    }

The API returns:

- Patient input
- Rule-based/KRR findings
- Predictions from all ML models
- Diabetes probability
- Hybrid risk score
- Overall risk level
- ML prediction
- Recommendations

Example response structure:

    {
      "patient": {},
      "krr": {
        "risk_factors": [],
        "conditions": [],
        "inferred_classes": [],
        "is_high_risk": false
      },
      "ml_models": {
        "Random Forest": {},
        "Logistic Regression": {},
        "Gradient Boosting": {},
        "SVM": {}
      },
      "hybrid_score": 62.5,
      "risk_level": "Moderate",
      "ml_prediction": 1,
      "recommendations": []
    }

--------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------

Recommendations are generated according to the hybrid risk score and detected rule-based conditions.

The system can generate recommendations related to:

- Clinical follow-up
- HbA1c testing
- Fasting glucose retesting
- Lifestyle modifications
- Weight management
- Blood pressure monitoring
- Glucose monitoring
- Pre-diabetic indicators

These recommendations are intended for educational demonstration only and should not be interpreted as personalized medical advice.

--------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------

A possible repository structure:

    GlucoSense/
    |
    +-- backend/
    |   +-- app.py
    |   +-- diabetes.csv
    |   +-- requirements.txt
    |
    +-- frontend/
    |   +-- index.html
    |   +-- style.css
    |   +-- script.js
    |
    +-- ImageClassification.ipynb
    |
    +-- README.txt
    +-- ...

Adapt the structure to match the actual files in the repository.

--------------------------------------------------
TECHNOLOGIES
--------------------------------------------------

Python
Flask
Flask-CORS
NumPy
Pandas
Scikit-learn
Imbalanced-learn
Random Forest
Logistic Regression
Gradient Boosting
Support Vector Machine
StandardScaler
HTML
CSS
JavaScript

--------------------------------------------------
INSTALLATION
--------------------------------------------------

1. Clone the repository:

    git clone https://github.com/sharifalkadi/GlucoSense.git
    cd GlucoSense

2. Create a virtual environment.

Windows:

    python -m venv venv
    venv\Scripts\activate

Linux/macOS:

    python3 -m venv venv
    source venv/bin/activate

3. Install dependencies:

    pip install flask flask-cors numpy pandas scikit-learn imbalanced-learn

Or, if requirements.txt is available:

    pip install -r requirements.txt

--------------------------------------------------
RUNNING THE APPLICATION
--------------------------------------------------

Make sure diabetes.csv is located in the expected directory.

Start the Flask server:

    python app.py

The API will be available at:

    http://localhost:5000

The terminal should display:

    Diabetes Detection API starting...
    Open index.html in your browser to use the UI

--------------------------------------------------
TESTING THE API
--------------------------------------------------

Health check:

    GET http://localhost:5000/api/health

Prediction:

    POST http://localhost:5000/api/predict

Send a JSON body containing the eight patient features.

--------------------------------------------------
ERROR HANDLING
--------------------------------------------------

If no JSON body is provided:

    {
      "error": "No JSON body"
    }

If an unexpected server-side error occurs:

    {
      "error": "..."
    }

The application also includes a fallback/demo mode if the dataset cannot be loaded.

--------------------------------------------------
IMPORTANT IMPLEMENTATION NOTES
--------------------------------------------------

DEMO MODE

If diabetes.csv cannot be loaded, the application switches to demo mode.

In demo mode, the API generates random prediction probabilities.

Demo mode should NOT be used for actual evaluation or medical applications.

MODEL SELECTION

Although four models are trained, the current hybrid scoring system uses the Random Forest model as the primary ML prediction.

The other models are still returned by the API for comparison.

CLINICAL INTERPRETATION

The thresholds used in the rule-based system are simplified project rules rather than a complete clinical decision system.

Real-world clinical decision support would require medically validated thresholds, appropriate preprocessing, external validation, and clinical oversight.

--------------------------------------------------
FUTURE IMPROVEMENTS
--------------------------------------------------

Possible improvements include:

- Hyperparameter optimization
- Cross-validation
- More comprehensive model evaluation
- ROC-AUC analysis
- Precision-recall analysis
- Confusion matrices
- Feature importance analysis
- Explainable AI
- Improved missing-value handling
- Clinical rule validation
- Ontology-based KRR using RDF/OWL
- Integration of a proper reasoning engine
- Model persistence instead of retraining at every server startup
- Secure API authentication
- Database integration
- Docker deployment
- Cloud deployment
- Automated testing
- Improved frontend visualization
- Model monitoring and logging

--------------------------------------------------
PROJECT OBJECTIVES
--------------------------------------------------

The main objective of GlucoSense is to demonstrate how machine learning and knowledge-based reasoning can be combined into a single AI system.

The project demonstrates:

- Supervised machine learning
- Binary classification
- Ensemble learning
- Feature scaling
- Class imbalance handling
- SMOTE
- Probability-based prediction
- Rule-based reasoning
- Hybrid AI systems
- REST API development
- Frontend/backend integration
- AI-assisted decision support

--------------------------------------------------
AUTHOR
--------------------------------------------------

Sharif Alkadi

Bachelor of Artificial Intelligence

GitHub:
https://github.com/sharifalkadi

--------------------------------------------------
DISCLAIMER
--------------------------------------------------

GlucoSense is an academic and educational AI project.

It has not been validated for clinical use and should not be used to diagnose diabetes, determine treatment, or make medical decisions.

Always consult a qualified healthcare professional for medical diagnosis and treatment.

--------------------------------------------------
LICENSE
--------------------------------------------------

This project is intended primarily for educational and research purposes.
