from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class LoanFeatures(BaseModel):
    # --- Informasi pinjaman ---
    loan_amnt: Optional[float] = Field(13000, description="Jumlah pinjaman yang diajukan (USD)")
    term: Optional[float] = Field(36, description="Tenor pinjaman dalam bulan (36 atau 60)")
    int_rate: Optional[float] = Field(12.99, description="Suku bunga pinjaman (%)")
    installment: Optional[float] = Field(382.55, description="Cicilan bulanan (USD)")
    sub_grade_enc: Optional[float] = Field(10, description="Sub-grade LendingClub terenkode ordinal (A1=0 ... G5=34)")
    verification_status_enc: Optional[float] = Field(1, description="Status verifikasi income: 0=Not Verified, 1=Source Verified, 2=Verified")

    # --- Profil peminjam ---
    emp_length: Optional[float] = Field(6, description="Lama bekerja dalam tahun (0.5=<1 tahun, 10=10+ tahun)")
    annual_inc: Optional[float] = Field(65000, description="Pendapatan tahunan (USD)")
    dti: Optional[float] = Field(17.66, description="Debt-to-Income ratio (%)")
    credit_history_months: Optional[float] = Field(178.9, description="Umur riwayat kredit dalam bulan")

    # --- Riwayat kredit ---
    inq_last_6mths: Optional[float] = Field(0, description="Jumlah inquiry kredit 6 bulan terakhir")
    delinq_2yrs: Optional[float] = Field(0, description="Jumlah keterlambatan bayar (>=30 hari) dalam 2 tahun terakhir")
    mths_since_last_delinq: Optional[float] = Field(31, description="Bulan sejak keterlambatan bayar terakhir")
    pub_rec: Optional[float] = Field(0, description="Jumlah catatan publik negatif (pailit, dll.)")
    open_acc: Optional[float] = Field(11, description="Jumlah akun kredit yang masih aktif/terbuka")
    total_acc: Optional[float] = Field(24, description="Total jumlah akun kredit yang pernah dimiliki")
    revol_util: Optional[float] = Field(56.0, description="Persentase penggunaan revolving credit line (%)")
    tot_cur_bal: Optional[float] = Field(80766, description="Total saldo seluruh akun saat ini (USD)")
    tot_coll_amt: Optional[float] = Field(0, description="Total jumlah tunggakan yang pernah masuk collection (USD)")
    total_rev_hi_lim: Optional[float] = Field(23700, description="Total limit tertinggi revolving credit (USD)")
    collections_12_mths_ex_med: Optional[float] = Field(0, description="Jumlah collection 12 bulan terakhir (di luar medis)")
    acc_now_delinq: Optional[float] = Field(0, description="Jumlah akun yang saat ini menunggak (delinquent)")

    # --- Kepemilikan rumah (one-hot; kategori dasar = OTHER/NONE/ANY) ---
    home_ownership_MORTGAGE: Optional[float] = Field(1, description="1 jika status kepemilikan rumah = MORTGAGE, else 0")
    home_ownership_RENT: Optional[float] = Field(0, description="1 jika status kepemilikan rumah = RENT, else 0")
    home_ownership_OWN: Optional[float] = Field(0, description="1 jika status kepemilikan rumah = OWN, else 0")

    # --- Tujuan pinjaman (one-hot; kategori dasar = 'other') ---
    purpose_debt_consolidation: Optional[float] = Field(1, description="1 jika tujuan pinjaman = debt_consolidation, else 0")
    purpose_credit_card: Optional[float] = Field(0, description="1 jika tujuan pinjaman = credit_card, else 0")
    purpose_home_improvement: Optional[float] = Field(0, description="1 jika tujuan pinjaman = home_improvement, else 0")
    purpose_major_purchase: Optional[float] = Field(0, description="1 jika tujuan pinjaman = major_purchase, else 0")
    purpose_small_business: Optional[float] = Field(0, description="1 jika tujuan pinjaman = small_business, else 0")
    purpose_car: Optional[float] = Field(0, description="1 jika tujuan pinjaman = car, else 0")
    purpose_wedding: Optional[float] = Field(0, description="1 jika tujuan pinjaman = wedding, else 0")
    purpose_medical: Optional[float] = Field(0, description="1 jika tujuan pinjaman = medical, else 0")
    purpose_house: Optional[float] = Field(0, description="1 jika tujuan pinjaman = house, else 0")

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 13000,
                "term": 36,
                "int_rate": 12.99,
                "installment": 382.55,
                "sub_grade_enc": 10,
                "verification_status_enc": 1,
                "emp_length": 6,
                "annual_inc": 65000,
                "dti": 17.66,
                "credit_history_months": 178.9,
                "inq_last_6mths": 0,
                "delinq_2yrs": 0,
                "mths_since_last_delinq": 31,
                "pub_rec": 0,
                "open_acc": 11,
                "total_acc": 24,
                "revol_util": 56.0,
                "tot_cur_bal": 80766,
                "tot_coll_amt": 0,
                "total_rev_hi_lim": 23700,
                "collections_12_mths_ex_med": 0,
                "acc_now_delinq": 0,
                "home_ownership_MORTGAGE": 1,
                "home_ownership_RENT": 0,
                "home_ownership_OWN": 0,
                "purpose_debt_consolidation": 1,
                "purpose_credit_card": 0,
                "purpose_home_improvement": 0,
                "purpose_major_purchase": 0,
                "purpose_small_business": 0,
                "purpose_car": 0,
                "purpose_wedding": 0,
                "purpose_medical": 0,
                "purpose_house": 0,
            }
        }


class ClassProbability(BaseModel):
    class_id: int
    label: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_label: str
    probabilities: List[ClassProbability]
    imputed_fields: List[str] = Field(
        default_factory=list,
        description="Daftar nama fitur yang nilainya kosong dan diisi otomatis dengan median training",
    )


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class ShapResponse(BaseModel):
    predicted_class_id: int
    predicted_label: str
    base_value: float
    contributions: List[FeatureContribution]
    imputed_fields: List[str] = Field(default_factory=list)


class PredictWithShapResponse(BaseModel):
    prediction: PredictionResponse
    shap: ShapResponse


class ModelInfoResponse(BaseModel):
    best_experiment_id: str
    description: str
    config: Dict
    n_features: int
    features: List[str]
    label_names: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
