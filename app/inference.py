import json
import logging
import threading

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from app import config
from app import preprocessing

logger = logging.getLogger("inference")


class InferenceEngine:
    _lock = threading.Lock()

    def __init__(self):
        logger.info("Memuat artifact & model...")

        self.model = tf.keras.models.load_model(config.MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.imputer = joblib.load(config.IMPUTER_PATH)

        with open(config.SELECTED_FEATURES_PATH) as f:
            self.selected_features = json.load(f)

        with open(config.LABEL_NAMES_PATH) as f:
            raw_label_names = json.load(f)
        self.label_names = {int(k): v for k, v in raw_label_names.items()}

        with open(config.BEST_MODEL_INFO_PATH) as f:
            self.best_model_info = json.load(f)

        with open(config.SUB_GRADE_MAPPING_PATH) as f:
            self.sub_grade_map = json.load(f)

        # Validasi urutan fitur scaler == selected_features (harus identik)
        scaler_features = list(getattr(self.scaler, "feature_names_in_", self.selected_features))
        if scaler_features != self.selected_features:
            raise RuntimeError(
                "Urutan fitur scaler_v2.pkl tidak sama dengan selected_features_v2.json. "
                "Periksa kembali kecocokan artifact."
            )

        # Median training per fitur (untuk isi nilai kosong / imputasi)
        imputer_features = list(self.imputer.feature_names_in_)
        stats = dict(zip(imputer_features, self.imputer.statistics_))
        missing = [f for f in self.selected_features if f not in stats]
        if missing:
            raise RuntimeError(f"Fitur berikut tidak ada di imputer_v2.pkl: {missing}")
        self.medians = {f: float(stats[f]) for f in self.selected_features}

        self.n_features = len(self.selected_features)
        self.n_classes = len(self.label_names)

        self._explainer = None  # lazy init, dibuat saat SHAP pertama kali dipakai
        logger.info(
            "Model & artifact berhasil dimuat: %d fitur, %d kelas (%s)",
            self.n_features, self.n_classes, self.best_model_info.get("best_exp_id"),
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def build_feature_vector(self, payload: dict):
        """
        Ubah dict input (bisa mengandung None) menjadi array numpy (1, n_features)
        sesuai urutan selected_features, dengan nilai kosong diisi median training.

        Returns: (raw_vector: np.ndarray shape (1, n_features), imputed_fields: list[str])
        """
        imputed_fields = []
        values = []
        for feat in self.selected_features:
            v = payload.get(feat)
            if v is None:
                v = self.medians[feat]
                imputed_fields.append(feat)
            values.append(float(v))
        raw_vector = np.array(values, dtype=np.float64).reshape(1, -1)
        return raw_vector, imputed_fields

    def scale(self, raw_vector: np.ndarray) -> np.ndarray:
        raw_df = pd.DataFrame(raw_vector, columns=self.selected_features)
        return self.scaler.transform(raw_df)

    # ------------------------------------------------------------------
    # Prediksi
    # ------------------------------------------------------------------
    def predict_proba(self, raw_vector: np.ndarray) -> np.ndarray:
        scaled = self.scale(raw_vector)
        proba = self.model.predict(scaled, verbose=0)
        return proba[0]

    def predict(self, payload: dict):
        raw_vector, imputed_fields = self.build_feature_vector(payload)
        proba = self.predict_proba(raw_vector)
        pred_class = int(np.argmax(proba))
        return {
            "predicted_class_id": pred_class,
            "predicted_label": self.label_names[pred_class],
            "probabilities": [
                {"class_id": i, "label": self.label_names[i], "probability": float(p)}
                for i, p in enumerate(proba)
            ],
            "imputed_fields": imputed_fields,
        }

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------
    def _predict_fn_raw_space(self, x: np.ndarray) -> np.ndarray:
        """predict_fn yang menerima input dalam skala ASLI (belum di-scale),
        melakukan scaling internal, lalu mengembalikan probabilitas model.
        Dengan begitu SHAP value dihitung & diinterpretasikan dalam satuan
        fitur asli (lebih mudah dibaca frontend/pengguna)."""
        x = np.asarray(x, dtype=np.float64)
        x_df = pd.DataFrame(x, columns=self.selected_features)
        scaled = self.scaler.transform(x_df)
        return self.model.predict(scaled, verbose=0)

    def _build_background(self) -> np.ndarray:
        """Sintesis background dari Normal(mean_, scale_) milik scaler
        (lihat catatan desain di docstring modul ini)."""
        rng = np.random.default_rng(config.SHAP_RANDOM_STATE)
        mean = self.scaler.mean_
        std = self.scaler.scale_
        n = config.SHAP_BACKGROUND_SIZE
        samples = rng.normal(loc=mean, scale=std, size=(n, len(mean)))

        # Clip fitur biner (one-hot / ordinal encoded 0/1) supaya background
        # tetap masuk akal (tidak generate nilai negatif utk fitur non-negatif)
        non_negative_like = [
            i for i, f in enumerate(self.selected_features)
            if f.startswith(("home_ownership_", "purpose_"))
            or f in ("inq_last_6mths", "delinq_2yrs", "pub_rec", "open_acc", "total_acc",
                      "tot_coll_amt", "collections_12_mths_ex_med", "acc_now_delinq",
                      "annual_inc", "loan_amnt", "installment", "revol_util",
                      "tot_cur_bal", "total_rev_hi_lim", "credit_history_months",
                      "term", "verification_status_enc", "sub_grade_enc", "emp_length")
        ]
        for i in non_negative_like:
            samples[:, i] = np.clip(samples[:, i], a_min=0, a_max=None)

        return samples

    def _get_explainer(self):
        if self._explainer is None:
            with self._lock:
                if self._explainer is None:
                    logger.info("Inisialisasi SHAP explainer (sekali saja)...")
                    background = self._build_background()
                    self._explainer = shap.Explainer(
                        self._predict_fn_raw_space,
                        background,
                        feature_names=self.selected_features,
                    )
        return self._explainer

    def explain(self, payload: dict):
        raw_vector, imputed_fields = self.build_feature_vector(payload)
        proba = self.predict_proba(raw_vector)
        pred_class = int(np.argmax(proba))

        explainer = self._get_explainer()
        shap_values = explainer(raw_vector)

        # shap_values.values shape: (1, n_features, n_classes) untuk output multi-kelas
        values = shap_values.values[0]
        if values.ndim == 2:
            class_values = values[:, pred_class]
            base_value = float(np.array(shap_values.base_values[0]).reshape(-1)[pred_class])
        else:
            class_values = values
            base_value = float(np.array(shap_values.base_values).reshape(-1)[0])

        contributions = [
            {
                "feature": feat,
                "value": float(raw_vector[0, i]),
                "shap_value": float(class_values[i]),
            }
            for i, feat in enumerate(self.selected_features)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        return {
            "predicted_class_id": pred_class,
            "predicted_label": self.label_names[pred_class],
            "base_value": base_value,
            "contributions": contributions,
            "imputed_fields": imputed_fields,
        }

    # ------------------------------------------------------------------
    # Entry point untuk input MENTAH (raw loan application fields)
    # ------------------------------------------------------------------
    def encode_raw(self, raw_payload: dict) -> dict:
        """Transformasi field mentah -> 34 fitur ter-encode, mereplikasi
        logika notebook 01b/02b (lihat app/preprocessing.py)."""
        return preprocessing.transform_raw_to_encoded(raw_payload, self.sub_grade_map)

    def predict_from_raw(self, raw_payload: dict):
        encoded = self.encode_raw(raw_payload)
        return self.predict(encoded)

    def explain_from_raw(self, raw_payload: dict):
        encoded = self.encode_raw(raw_payload)
        return self.explain(encoded)

    def model_info(self):
        return {
            "best_experiment_id": self.best_model_info.get("best_exp_id"),
            "description": self.best_model_info.get("config", {}).get("description", ""),
            "config": self.best_model_info.get("config", {}),
            "n_features": self.n_features,
            "features": self.selected_features,
            "label_names": {str(k): v for k, v in self.label_names.items()},
        }


# Singleton instance, dimuat sekali saat modul pertama kali diimpor (startup app)
engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global engine
    if engine is None:
        engine = InferenceEngine()
    return engine
