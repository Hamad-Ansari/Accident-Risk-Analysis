"""
RoadSafe AI - Model Training & Artifact Generator
Trains ML Risk Classification Models (Logistic Regression, Random Forest, XGBoost)
and exports version-compatible model artifacts for the Streamlit deployment.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Optional XGBoost import with graceful fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def build_synthetic_traffic_dataset(n_samples=50000):
    """
    Generates a realistic, domain-grounded synthetic road safety dataset
    matching standard NHTSA and US-Accidents risk factors.
    Used for instant self-contained model generation across any platform.
    """
    np.random.seed(RANDOM_STATE)
    
    weather_options = ['Clear', 'Rain', 'Fog', 'Snow', 'Overcast', 'Severe Storm']
    weather_probs = [0.45, 0.25, 0.10, 0.10, 0.07, 0.03]
    weather = np.random.choice(weather_options, size=n_samples, p=weather_probs)
    
    road_conditions = ['Dry', 'Wet', 'Icy', 'Snow Covered', 'Standing Water']
    road_cond_probs = [0.55, 0.25, 0.08, 0.08, 0.04]
    road_condition = np.random.choice(road_conditions, size=n_samples, p=road_cond_probs)
    
    time_of_day = np.random.choice(['Morning Rush', 'Daytime', 'Evening Rush', 'Night/Late'], size=n_samples, p=[0.25, 0.35, 0.25, 0.15])
    sunrise_sunset = np.where(time_of_day == 'Night/Late', 'Night', 'Day')
    
    visibility = np.clip(np.random.normal(loc=8.5, scale=2.5, size=n_samples), 0.1, 10.0)
    # Lower visibility when fog/rain/snow
    for i in range(n_samples):
        if weather[i] in ['Fog', 'Severe Storm']:
            visibility[i] = np.random.uniform(0.1, 2.5)
        elif weather[i] in ['Rain', 'Snow']:
            visibility[i] = np.random.uniform(1.0, 5.0)
            
    temperature = np.random.normal(loc=65.0, scale=18.0, size=n_samples)
    humidity = np.clip(np.random.normal(loc=60.0, scale=20.0, size=n_samples), 10.0, 100.0)
    wind_speed = np.clip(np.random.exponential(scale=7.0, size=n_samples), 0.0, 55.0)
    
    traffic_signal = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    junction = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    crossing = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    speed_limit = np.random.choice([25, 35, 45, 55, 65, 75], size=n_samples, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05])
    
    # Calculate Risk Score baseline
    risk_score = np.zeros(n_samples)
    
    # Weather impact
    weather_risk_map = {'Clear': 0.0, 'Overcast': 0.1, 'Rain': 0.4, 'Fog': 0.6, 'Snow': 0.7, 'Severe Storm': 0.9}
    risk_score += np.vectorize(weather_risk_map.get)(weather)
    
    # Road condition impact
    road_risk_map = {'Dry': 0.0, 'Wet': 0.35, 'Standing Water': 0.6, 'Snow Covered': 0.75, 'Icy': 0.95}
    risk_score += np.vectorize(road_risk_map.get)(road_condition)
    
    # Visibility impact
    risk_score += np.where(visibility < 2.0, 0.5, np.where(visibility < 5.0, 0.2, 0.0))
    
    # Night / Rush hour impact
    risk_score += np.where(time_of_day == 'Night/Late', 0.4, 0.0)
    risk_score += np.where(np.isin(time_of_day, ['Morning Rush', 'Evening Rush']), 0.25, 0.0)
    
    # Infrastructure & Speed
    risk_score += (speed_limit / 100.0) * 0.4
    risk_score += np.where(junction == 1, 0.25, 0.0)
    risk_score += np.where(wind_speed > 25, 0.3, 0.0)
    
    # Add random noise
    risk_score += np.random.normal(0, 0.2, size=n_samples)
    
    # Quantiles to define Low, Medium, High
    q_low = np.percentile(risk_score, 40)
    q_high = np.percentile(risk_score, 75)
    
    risk_level = np.where(risk_score < q_low, 'Low',
                          np.where(risk_score < q_high, 'Medium', 'High'))
    
    df = pd.DataFrame({
        'Weather_Condition': weather,
        'Road_Condition': road_condition,
        'Time_of_Day': time_of_day,
        'Sunrise_Sunset': sunrise_sunset,
        'Visibility(mi)': np.round(visibility, 1),
        'Temperature(F)': np.round(temperature, 1),
        'Humidity(%)': np.round(humidity, 1),
        'Wind_Speed(mph)': np.round(wind_speed, 1),
        'Speed_Limit': speed_limit,
        'Traffic_Signal': traffic_signal,
        'Junction': junction,
        'Crossing': crossing,
        'Risk_Level': risk_level
    })
    
    return df


def train_and_save_models():
    print("="*60)
    print("RoadSafe AI: Training Risk Classification Models...")
    print("="*60)
    
    df = build_synthetic_traffic_dataset(n_samples=40000)
    print(f"Generated dataset with {len(df)} records.")
    print("Risk Level Distribution:")
    print(df['Risk_Level'].value_counts(normalize=True))
    
    target_col = 'Risk_Level'
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    le = LabelEncoder()
    # Fit with explicit classes ordering: ['Low', 'Medium', 'High']
    le.fit(['Low', 'Medium', 'High'])
    y_encoded = le.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1)
    
    results = []
    fitted_pipelines = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': round(float(acc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1-score': round(float(f1), 4)
        })
        fitted_pipelines[name] = pipe
        print(f" -> {name}: Accuracy={acc:.4f}, F1-score={f1:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('F1-score', ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = fitted_pipelines[best_model_name]
    
    print(f"\n>>> Best Performing Model: {best_model_name}")
    
    # Detailed evaluation on best model
    best_preds = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_preds).tolist()
    class_names = le.classes_.tolist()
    report_dict = classification_report(y_test, best_preds, target_names=class_names, output_dict=True)
    
    # Save artifacts
    model_path = ARTIFACTS_DIR / "best_risk_model.joblib"
    encoder_path = ARTIFACTS_DIR / "risk_label_encoder.joblib"
    meta_path = ARTIFACTS_DIR / "model_metadata.json"
    
    joblib.dump(best_pipeline, model_path)
    joblib.dump(le, encoder_path)
    
    metadata = {
        'best_model_name': best_model_name,
        'feature_cols': feature_cols,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'class_names': class_names,
        'model_comparison': results,
        'confusion_matrix': cm,
        'classification_report': report_dict
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nSaved artifacts successfully to {ARTIFACTS_DIR.absolute()}:")
    print(f" - {model_path.name}")
    print(f" - {encoder_path.name}")
    print(f" - {meta_path.name}")
    print("="*60)


if __name__ == "__main__":
    train_and_save_models()
