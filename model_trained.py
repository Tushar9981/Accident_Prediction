import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. GENERATE A REALISTIC 1,000-ROW DATASET FOR TRAINING
np.random.seed(42)
n_samples = 1000

weathers = np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], size=n_samples, p=[0.5, 0.25, 0.15, 0.10])
road_conds = []
for w in weathers:
    if w == 'Clear':
        road_conds.append(np.random.choice(['Dry', 'Wet'], p=[0.9, 0.1]))
    elif w == 'Rain':
        road_conds.append(np.random.choice(['Wet', 'Dry'], p=[0.85, 0.15]))
    else:  # Fog or Snow
        road_conds.append(np.random.choice(['Icy', 'Wet', 'Dry'], p=[0.6, 0.3, 0.1]))

times_of_day = np.random.choice(['Day', 'Night'], size=n_samples, p=[0.6, 0.4])
speed_limits = np.random.choice([25, 35, 45, 55, 65, 70], size=n_samples)
driver_ages = np.random.randint(18, 75, size=n_samples)
visibility_km = np.random.uniform(0.5, 10.0, size=n_samples)

# REALISTIC RISK LOGIC FOR GROUND TRUTH LABELS
# Risk factors combine additively to create real non-linear risk weights
risk_score = (
    (np.array(weathers) == 'Snow') * 2.5 +
    (np.array(weathers) == 'Fog') * 2.0 +
    (np.array(weathers) == 'Rain') * 1.2 +
    (np.array(road_conds) == 'Icy') * 3.0 +
    (np.array(road_conds) == 'Wet') * 1.5 +
    (np.array(times_of_day) == 'Night') * 1.3 +
    (speed_limits / 20.0) +
    (driver_ages < 22) * 1.8 +
    (driver_ages > 68) * 1.2 +
    (10.0 - visibility_km) * 0.4
)

# Convert score to binary accident severity probability
prob = 1 / (1 + np.exp(-(risk_score - 7.5)))
y = (prob > 0.5).astype(int)

df = pd.DataFrame({
    'weather': weathers,
    'road_condition': road_conds,
    'time_of_day': times_of_day,
    'speed_limit': speed_limits,
    'driver_age': driver_ages,
    'visibility_km': visibility_km,
    'accident_severity': y
})

# 2. SEPARATE FEATURES & TARGET
X = df.drop(columns=['accident_severity'])
y = df['accident_severity']

# 3. BUILD ADVANCED PIPELINE
categorical_cols = ['weather', 'road_condition', 'time_of_day']
numerical_cols = ['speed_limit', 'driver_age', 'visibility_km']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# Gradient Boosting offers high prediction accuracy for complex tabular data
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    ))
])

# 4. TRAIN AND EVALUATE MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print(f"Model Training Complete!")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 5. SAVE HIGH-POWERED MODEL
joblib.dump(model_pipeline, 'accident_model.joblib')
print("Powerful new model saved to 'accident_model.joblib'!")
