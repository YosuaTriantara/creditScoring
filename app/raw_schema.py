from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal


class LoanApplicationRaw(BaseModel):
    # --- Informasi pinjaman ---
    loan_amnt: Optional[float] = Field(13000, description="Jumlah pinjaman yang diajukan (USD)")
    term: Optional[Literal["36 months", "60 months"]] = Field("36 months", description="Tenor pinjaman")
    int_rate: Optional[float] = Field(12.99, description="Suku bunga pinjaman (%), mis. 12.99 untuk 12.99%")
    installment: Optional[float] = Field(382.55, description="Cicilan bulanan (USD)")
    sub_grade: Optional[str] = Field("C1", description="Sub-grade LendingClub, format huruf+angka (A1..G5)")
    verification_status: Optional[Literal["Not Verified", "Source Verified", "Verified"]] = Field(
        "Source Verified", description="Status verifikasi pendapatan"
    )
    purpose: Optional[str] = Field(
        "debt_consolidation",
        description=(
            "Tujuan pinjaman. Salah satu dari: debt_consolidation, credit_card, "
            "home_improvement, major_purchase, small_business, car, wedding, medical, "
            "house, educational, moving, renewable_energy, vacation, other. "
            "Kategori di luar 9 kategori berpengaruh (lihat /model/info) akan "
            "diperlakukan setara dengan 'other' oleh model."
        ),
    )

    # --- Profil peminjam ---
    emp_length: Optional[str] = Field(
        "6 years",
        description="Lama bekerja, format asli: '< 1 year', '1 year'..'9 years', '10+ years'",
    )
    home_ownership: Optional[Literal["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"]] = Field(
        "MORTGAGE", description="Status kepemilikan rumah"
    )
    annual_inc: Optional[float] = Field(65000, description="Pendapatan tahunan (USD)")
    dti: Optional[float] = Field(17.66, description="Debt-to-Income ratio (%)")
    earliest_cr_line: Optional[str] = Field(
        "Jan-2010",
        description=(
            "Tanggal pembukaan jalur kredit pertama, format 'Mon-YYYY' (mis. 'Jan-2010'). "
            "Dipakai untuk menghitung umur riwayat kredit (credit_history_months) relatif "
            "terhadap tanggal pengajuan saat ini."
        ),
    )

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

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 13000,
                "term": "36 months",
                "int_rate": 12.99,
                "installment": 382.55,
                "sub_grade": "C1",
                "verification_status": "Source Verified",
                "purpose": "debt_consolidation",
                "emp_length": "6 years",
                "home_ownership": "MORTGAGE",
                "annual_inc": 65000,
                "dti": 17.66,
                "earliest_cr_line": "Jan-2010",
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
            }
        }
