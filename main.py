import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. SIMULATED DATA INGESTION
def load_data():
    # Simulating 20+ features as per your CV
    data = np.random.rand(1000, 20)
    columns = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(data, columns=columns)
    df['category'] = np.random.choice(['Type_A', 'Type_B'], size=1000)
    df['target'] = np.random.randint(0, 2, size=1000)
    return df

# 2. AUTOMATED FEATURE ENGINEERING
def engineer_features(df):
    # Example of extracting new intelligence from raw data
    df['feature_sum'] = df[[f'feature_{i}' for i in range(5)]].sum(axis=1)
    df['feature_mean'] = df[[f'feature_{i}' for i in range(5, 10)]].mean(axis=1)
    # This fulfills the '20+ features' claim in your CV
    return df

# 3. BUILDING THE PIPELINE
def build_pipeline(model_type='rf'):
    # Preprocessing for numeric and categorical data
    numeric_features = [f'feature_{i}' for i in range(20)] + ['feature_sum', 'feature_mean']
    categorical_features = ['category']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Choosing the algorithm based on project requirements
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    elif model_type == 'svm':
        clf = SVC(kernel='rbf', probability=True)
    else:
        clf = GradientBoostingClassifier()

    # The actual Pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

# 4. EXECUTION
if __name__ == "__main__":
    print("--- Initializing End-to-End ML Pipeline ---")
    
    # Process
    df = load_data()
    df = engineer_features(df)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train using Random Forest (as per CV)
    model_pipeline = build_pipeline(model_type='rf')
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    predictions = model_pipeline.predict(X_test)
    print("Model Evaluation Results:")
    print(classification_report(y_test, predictions))
    
    print("--- Pipeline Execution Complete ---")