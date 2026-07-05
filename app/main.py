import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import get_engine
from app.raw_schema import LoanApplicationRaw
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
        "hasil feature selection). Endpoint utama menerima data mentah form "
        "aplikasi pinjaman dan melakukan preprocessing lengkap secara internal "
        "(mereplikasi notebook 01b/02b). Menyediakan prediksi kelas risiko dan "
        "penjelasan SHAP per instance. Dibuat untuk kebutuhan testing lewat "
        "Swagger UI; integrasi frontend dikembangkan terpisah."
    ),
    version="1.1.0",
)

# CORS dibuka untuk semua origin - memudahkan testing dari frontend terpisah nantinya.
# Sesuaikan allow_origins jika sudah masuk tahap production.
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


# ======================================================================
# ENDPOINT UTAMA — input mentah (raw loan application fields)
# ======================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Inference (Raw Input)"])
def predict(application: LoanApplicationRaw):
    """
    Prediksi kelas risiko peminjam dari data MENTAH form aplikasi pinjaman.

    Preprocessing (parsing term/emp_length, encoding sub_grade/verification_status,
    one-hot purpose/home_ownership, perhitungan credit_history_months) dilakukan
    otomatis di dalam endpoint ini, mereplikasi notebook 01b & 02b.

    Field yang dikosongkan akan otomatis diisi median dari data training
    (untuk field numerik/ordinal) atau kategori baseline (untuk field kategorikal).

    Hasil kelas: 0 = Prime, 1 = Performing, 2 = Non-Performing.
    """
    engine = get_engine()
    try:
        raw_payload = application.model_dump()
        return engine.predict_from_raw(raw_payload)
    except Exception as e:
        logger.exception("Prediksi gagal")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {e}")


@app.post("/explain/shap", response_model=ShapResponse, tags=["Inference (Raw Input)"])
def explain_shap(application: LoanApplicationRaw):
    """
    Hitung SHAP value (local feature attribution) dari data MENTAH form aplikasi,
    untuk kelas yang diprediksi.

    Preprocessing sama seperti /predict. SHAP value dilaporkan dalam satuan
    fitur hasil encoding (34 fitur terpilih), diurutkan dari yang paling
    berpengaruh (nilai absolut).

    Catatan: karena SHAP menghitung banyak evaluasi model, endpoint ini
    lebih lambat dibanding /predict (perkiraan beberapa detik per request).
    """
    engine = get_engine()
    try:
        raw_payload = application.model_dump()
        return engine.explain_from_raw(raw_payload)
    except Exception as e:
        logger.exception("SHAP explain gagal")
        raise HTTPException(status_code=500, detail=f"Gagal menghitung SHAP: {e}")


@app.post("/predict/full", response_model=PredictWithShapResponse, tags=["Inference (Raw Input)"])
def predict_full(application: LoanApplicationRaw):
    """
    Endpoint gabungan: dari data MENTAH form aplikasi, mengembalikan hasil
    prediksi SEKALIGUS penjelasan SHAP dalam satu response (memudahkan
    konsumsi dari 1 endpoint untuk frontend nanti).
    """
    engine = get_engine()
    try:
        raw_payload = application.model_dump()
        encoded = engine.encode_raw(raw_payload)
        prediction = engine.predict(encoded)
        shap_result = engine.explain(encoded)
        return {"prediction": prediction, "shap": shap_result}
    except Exception as e:
        logger.exception("Predict+SHAP gagal")
        raise HTTPException(status_code=500, detail=f"Gagal memproses request: {e}")


# ======================================================================
# ENDPOINT LANJUTAN — input 34 fitur yang SUDAH ter-encode langsung
# (untuk debugging / konsumen yang sudah melakukan encoding sendiri)
# ======================================================================

@app.post("/predict/encoded", response_model=PredictionResponse, tags=["Inference (Encoded Features - Advanced)"])
def predict_encoded(features: LoanFeatures):
    """
    Sama seperti /predict, tapi menerima 34 fitur yang SUDAH ter-encode
    langsung (sesuai kolom scaler_v2.pkl) — tanpa preprocessing tambahan.
    Berguna untuk debugging atau jika encoding sudah dilakukan di sisi konsumen API.
    """
    engine = get_engine()
    try:
        payload = features.model_dump()
        return engine.predict(payload)
    except Exception as e:
        logger.exception("Prediksi (encoded) gagal")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {e}")


@app.post("/explain/shap/encoded", response_model=ShapResponse, tags=["Inference (Encoded Features - Advanced)"])
def explain_shap_encoded(features: LoanFeatures):
    """Sama seperti /explain/shap, tapi menerima 34 fitur yang sudah ter-encode langsung."""
    engine = get_engine()
    try:
        payload = features.model_dump()
        return engine.explain(payload)
    except Exception as e:
        logger.exception("SHAP explain (encoded) gagal")
        raise HTTPException(status_code=500, detail=f"Gagal menghitung SHAP: {e}")


@app.post("/predict/full/encoded", response_model=PredictWithShapResponse, tags=["Inference (Encoded Features - Advanced)"])
def predict_full_encoded(features: LoanFeatures):
    """Sama seperti /predict/full, tapi menerima 34 fitur yang sudah ter-encode langsung."""
    engine = get_engine()
    try:
        payload = features.model_dump()
        prediction = engine.predict(payload)
        shap_result = engine.explain(payload)
        return {"prediction": prediction, "shap": shap_result}
    except Exception as e:
        logger.exception("Predict+SHAP (encoded) gagal")
        raise HTTPException(status_code=500, detail=f"Gagal memproses request: {e}")
