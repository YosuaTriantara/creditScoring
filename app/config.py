import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model_v2.keras")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler_v2.pkl")
IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, "imputer_v2.pkl")

SELECTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "selected_features_v2.json")
LABEL_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "label_names_v2.json")
BEST_MODEL_INFO_PATH = os.path.join(ARTIFACTS_DIR, "best_model_info_v2.json")

SHAP_BACKGROUND_SIZE = 50
SHAP_RANDOM_STATE = 42
