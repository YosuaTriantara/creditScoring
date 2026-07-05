import re
from datetime import datetime

# --- Mapping identik dengan 02b_Feature_Engineering.ipynb ---
VERIFICATION_STATUS_MAP = {
    "Not Verified": 0,
    "Source Verified": 1,
    "Verified": 2,
}

PURPOSE_COLUMNS = [
    "debt_consolidation", "credit_card", "small_business", "wedding",
    "major_purchase", "car", "house", "medical", "home_improvement",
]

HOME_OWNERSHIP_COLUMNS = ["MORTGAGE", "RENT", "OWN"]

DIRECT_NUMERIC_FIELDS = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq", "open_acc",
    "pub_rec", "revol_util", "total_acc", "tot_coll_amt", "tot_cur_bal",
    "total_rev_hi_lim", "collections_12_mths_ex_med", "acc_now_delinq",
]


def parse_term(term_raw):
    """'36 months' -> 36.0  (identik dengan regex di 01b)"""
    if term_raw is None:
        return None
    m = re.search(r"(\d+)", str(term_raw))
    return float(m.group(1)) if m else None


def parse_emp_length(emp_length_raw):
    """Identik dengan fungsi parse_emp_length() di 01b_Data_Preprocessing.ipynb."""
    if emp_length_raw is None:
        return None
    s = str(emp_length_raw)
    if "<" in s:
        return 0.5
    if "10+" in s:
        return 10.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else None


def compute_credit_history_months(earliest_cr_line_raw, as_of=None):
    """
    Menghitung jumlah bulan riwayat kredit berdasarkan tanggal pembukaan rekening terlama.
    """
    if not earliest_cr_line_raw:
        return None
    as_of = as_of or datetime.now()
    try:
        earliest_dt = datetime.strptime(str(earliest_cr_line_raw).strip(), "%b-%Y")
    except ValueError:
        return None
    months = (as_of - earliest_dt).days / 30.44
    return round(months, 1)


def encode_sub_grade(sub_grade_raw, sub_grade_map: dict):
    if not sub_grade_raw:
        return None
    return sub_grade_map.get(str(sub_grade_raw).strip().upper())


def encode_verification_status(verification_status_raw):
    if not verification_status_raw:
        return None
    return VERIFICATION_STATUS_MAP.get(verification_status_raw)


def one_hot_purpose(purpose_raw):
    """Selalu mengembalikan 0/1 eksplisit (tidak pernah None) untuk tiap kolom
    purpose_* di 34 fitur terpilih. Kategori tak dikenal / None -> semua 0
    (setara kategori baseline 'other')."""
    purpose_norm = (purpose_raw or "").strip().lower()
    return {f"purpose_{p}": (1.0 if purpose_norm == p else 0.0) for p in PURPOSE_COLUMNS}


def one_hot_home_ownership(home_ownership_raw):
    """Sama seperti one_hot_purpose, tapi untuk home_ownership_*."""
    ho_norm = (home_ownership_raw or "").strip().upper()
    return {
        f"home_ownership_{h}": (1.0 if ho_norm == h else 0.0)
        for h in HOME_OWNERSHIP_COLUMNS
    }


def transform_raw_to_encoded(raw: dict, sub_grade_map: dict) -> dict:
    encoded = {}

    for field in DIRECT_NUMERIC_FIELDS:
        encoded[field] = raw.get(field)

    encoded["term"] = parse_term(raw.get("term"))
    encoded["emp_length"] = parse_emp_length(raw.get("emp_length"))
    encoded["credit_history_months"] = compute_credit_history_months(raw.get("earliest_cr_line"))
    encoded["sub_grade_enc"] = encode_sub_grade(raw.get("sub_grade"), sub_grade_map)
    encoded["verification_status_enc"] = encode_verification_status(raw.get("verification_status"))

    encoded.update(one_hot_purpose(raw.get("purpose")))
    encoded.update(one_hot_home_ownership(raw.get("home_ownership")))

    return encoded
