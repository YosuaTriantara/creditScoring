
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import get_engine
from app.schemas import (
    LoanFeatures,
    PredictionResponse,
    ShapResponse,
    PredictWithShapResponse,
    ModelInfoResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="Loan Risk Classification API",
    description=(
        "API inference untuk model klasifikasi risiko peminjam (ANN, 34 fitur "
        "hasil feature selection). Menyediakan prediksi kelas risiko dan "
        "penjelasan SHAP per instance. Dibuat untuk kebutuhan testing lewat "
        "Swagger UI; integrasi frontend dikembangkan terpisah."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model_on_startup():
    # Memuat model & artifact sekali di awal, supaya request pertama tidak lambat.
    get_engine()
    logger.info("Startup selesai: model & artifact siap dipakai.")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Cek status API dan apakah model sudah termuat."""
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "model_loaded": engine.model is not None,
            "n_features": engine.n_features,
        }
    except Exception as e:
        logger.exception("Health check gagal")
        raise HTTPException(status_code=503, detail=f"Model belum siap: {e}")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Metadata model terbaik: id eksperimen, konfigurasi, daftar fitur, dan mapping label."""
    engine = get_engine()
    return engine.model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: LoanFeatures):
    
    engine = get_engine()
    try:
        payload = features.model_dump()
        return engine.predict(payload)
    except Exception as e:
        logger.exception("Prediksi gagal")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {e}")


@app.post("/explain/shap", response_model=ShapResponse, tags=["Inference"])
def explain_shap(features: LoanFeatures):
    
    engine = get_engine()
    try:
        payload = features.model_dump()
        return engine.explain(payload)
    except Exception as e:
        logger.exception("SHAP explain gagal")
        raise HTTPException(status_code=500, detail=f"Gagal menghitung SHAP: {e}")


@app.post("/predict/full", response_model=PredictWithShapResponse, tags=["Inference"])
def predict_full(features: LoanFeatures):
    
    engine = get_engine()
    try:
        payload = features.model_dump()
        prediction = engine.predict(payload)
        shap_result = engine.explain(payload)
        return {"prediction": prediction, "shap": shap_result}
    except Exception as e:
        logger.exception("Predict+SHAP gagal")
        raise HTTPException(status_code=500, detail=f"Gagal memproses request: {e}")
