import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import mlflow
from huggingface_hub import login, HfApi


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/ananyaarvindiyer/tourism-package-prediction-ui/Xtrain.csv"
Xtest_path = "hf://datasets/ananyaarvindiyer/tourism-package-prediction-ui/Xtest.csv"
ytrain_path = "hf://datasets/ananyaarvindiyer/tourism-package-prediction-ui/ytrain.csv"
ytest_path = "hf://datasets/ananyaarvindiyer/tourism-package-prediction-ui/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define numeric and categorical features for preprocessing and better data handling of mixed dataset
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
    'ProductPitched', 'Designation'
]

# Preprocessor (no scaling needed for XGB)
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    remainder='passthrough'
)

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__subsample': [0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.8, 1.0]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=cv_strategy,
        n_jobs=-1,
        scoring='average_precision'
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # ---- Predictions ----
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    # ---- Metrics ----
    test_accuracy = accuracy_score(ytest, y_pred_test)
    test_precision = precision_score(ytest, y_pred_test)
    test_recall = recall_score(ytest, y_pred_test)
    test_f1 = f1_score(ytest, y_pred_test)
    test_roc_auc = roc_auc_score(ytest, y_proba_test)
    test_pr_auc = average_precision_score(ytest, y_proba_test)

    mlflow.log_metrics({
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "cv_best_average_precision": grid_search.best_score_
    })


    # Save the model locally
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "ananyaarvindiyer/tourism-package-prediction-ui"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_package_prediction_model_v1.joblib",
        path_in_repo="best_tourism_package_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
      )
