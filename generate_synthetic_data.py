"""
generate_synthetic_data.py

Generates FULLY SYNTHETIC, intentionally messy healthcare-style data for a
data cleaning portfolio project. No real patient data is used or referenced.

Outputs (in ./data/):
  patients_raw.csv      - messy patient demographics
  encounters_raw.csv    - messy visit/encounter records
  patients_clean.csv    - the "answer key" (before messiness was added)
  encounters_clean.csv  - the "answer key" for encounters

Usage:
  pip install faker pandas numpy
  python generate_synthetic_data.py --patients 2000 --seed 42
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------
ICD10_CODES = [
    ("E11.9", "Type 2 diabetes without complications"),
    ("I10", "Essential hypertension"),
    ("J45.909", "Unspecified asthma"),
    ("M54.5", "Low back pain"),
    ("F41.9", "Anxiety disorder, unspecified"),
    ("J06.9", "Acute upper respiratory infection"),
    ("K21.9", "Gastro-esophageal reflux disease"),
    ("Z00.00", "General adult medical exam"),
    ("E78.5", "Hyperlipidemia, unspecified"),
    ("N39.0", "Urinary tract infection"),
]

INSURERS = ["Blue Cross", "Aetna", "United Healthcare", "Cigna", "Medicare", "Medicaid", "Self-Pay"]
VISIT_TYPES = ["Office Visit", "Telehealth", "Urgent Care", "Follow-up", "Annual Physical"]
PROVIDERS = [f"PRV{n:03d}" for n in range(1, 16)]

STATE_FULL = {
    "GA": "Georgia", "FL": "Florida", "AL": "Alabama",
    "SC": "South Carolina", "TN": "Tennessee", "NC": "North Carolina",
}


# ---------------------------------------------------------------------------
# Step 1: build CLEAN data (the answer key)
# ---------------------------------------------------------------------------
def build_clean_patients(n: int, fake: Faker) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        sex = random.choice(["F", "M"])
        first = fake.first_name_female() if sex == "F" else fake.first_name_male()
        state = random.choice(list(STATE_FULL))
        rows.append({
            "patient_id": f"PT{i:06d}",
            "first_name": first,
            "last_name": fake.last_name(),
            "sex": sex,
            "date_of_birth": fake.date_of_birth(minimum_age=1, maximum_age=95),
            "phone": fake.numerify("(###) ###-####"),
            "email": fake.free_email(),
            "city": fake.city(),
            "state": state,
            "zip_code": fake.numerify("#####"),
            "insurance": random.choices(INSURERS, weights=[20, 18, 18, 14, 14, 10, 6])[0],
            "height_cm": round(np.random.normal(168 if sex == "F" else 178, 8), 1),
            "weight_kg": round(np.random.normal(72 if sex == "F" else 86, 14), 1),
        })
    df = pd.DataFrame(rows)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    return df


def build_clean_encounters(patients: pd.DataFrame, fake: Faker) -> pd.DataFrame:
    rows, enc_id = [], 1
    for _, p in patients.iterrows():
        for _ in range(np.random.poisson(2.5)):
            code, desc = random.choice(ICD10_CODES)
            visit_date = fake.date_between(start_date="-2y", end_date="today")
            rows.append({
                "encounter_id": f"ENC{enc_id:07d}",
                "patient_id": p["patient_id"],
                "provider_id": random.choice(PROVIDERS),
                "visit_date": pd.Timestamp(visit_date),
                "visit_type": random.choice(VISIT_TYPES),
                "diagnosis_code": code,
                "diagnosis_desc": desc,
                "systolic_bp": int(np.random.normal(124, 14)),
                "diastolic_bp": int(np.random.normal(79, 9)),
                "heart_rate": int(np.random.normal(75, 10)),
                "charge_amount": round(random.uniform(75, 650), 2),
                "no_show": random.random() < 0.12,
            })
            enc_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: inject realistic messiness
# ---------------------------------------------------------------------------
def pick(df: pd.DataFrame, frac: float) -> pd.Index:
    """Random subset of row labels."""
    return df.sample(frac=frac).index


def mess_up_patients(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()

    # Dates of birth in mixed formats (stored as text)
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%y", "%B %d, %Y", "%d/%m/%Y"]
    df["date_of_birth"] = df["date_of_birth"].apply(
        lambda d: d.strftime(random.choice(formats))
    )

    # Inconsistent name casing and stray whitespace
    for col in ["first_name", "last_name", "city"]:
        idx = pick(df, 0.15)
        df.loc[idx, col] = df.loc[idx, col].apply(
            lambda s: random.choice([s.upper(), s.lower(), f"  {s} ", f"{s} "])
        )

    # Sex coded many ways, plus some blanks
    sex_map = {"F": ["F", "Female", "female", "f", "FEMALE"],
               "M": ["M", "Male", "male", "m", "MALE"]}
    df["sex"] = df["sex"].apply(lambda s: random.choice(sex_map[s]))
    df.loc[pick(df, 0.02), "sex"] = np.nan
    df.loc[pick(df, 0.01), "sex"] = "Unknown"

    # Phone numbers in different formats
    def scramble_phone(p):
        digits = "".join(ch for ch in p if ch.isdigit())
        return random.choice([
            p, digits, f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
            f"{digits[:3]}.{digits[3:6]}.{digits[6:]}", f"1-{digits}",
        ])
    df["phone"] = df["phone"].apply(scramble_phone)
    df.loc[pick(df, 0.05), "phone"] = np.nan

    # State as abbreviation vs full name vs lowercase
    idx = pick(df, 0.25)
    df.loc[idx, "state"] = df.loc[idx, "state"].map(STATE_FULL)
    idx = pick(df, 0.10)
    df.loc[idx, "state"] = df.loc[idx, "state"].str.lower()

    # ZIP codes: missing, ZIP+4, or wrong length
    df["zip_code"] = df["zip_code"].astype(str)
    idx = pick(df, 0.08)
    df.loc[idx, "zip_code"] = df.loc[idx, "zip_code"] + "-" + [
        f"{random.randint(0, 9999):04d}" for _ in idx
    ]
    df.loc[pick(df, 0.03), "zip_code"] = np.nan
    df.loc[pick(df, 0.02), "zip_code"] = "0000"

    # Insurance name variants
    variants = {
        "Blue Cross": ["Blue Cross", "BCBS", "Blue Cross Blue Shield", "blue cross"],
        "United Healthcare": ["United Healthcare", "UHC", "United Health", "UnitedHealthcare"],
        "Self-Pay": ["Self-Pay", "Self Pay", "self pay", "None", "N/A"],
    }
    df["insurance"] = df["insurance"].apply(
        lambda v: random.choice(variants[v]) if v in variants else v
    )
    df.loc[pick(df, 0.04), "insurance"] = np.nan

    # Invalid emails
    idx = pick(df, 0.04)
    df.loc[idx, "email"] = df.loc[idx, "email"].str.replace("@", "", regex=False)
    df.loc[pick(df, 0.05), "email"] = np.nan

    # Mixed units and impossible values in height/weight
    df["weight_unit"] = "kg"
    idx = pick(df, 0.15)
    df.loc[idx, "weight_kg"] = (df.loc[idx, "weight_kg"] * 2.20462).round(1)
    df.loc[idx, "weight_unit"] = "lbs"
    df.loc[pick(df, 0.01), "height_cm"] = 0
    df.loc[pick(df, 0.01), "weight_kg"] = -1
    df.loc[pick(df, 0.01), "height_cm"] = 1680  # extra zero typo
    df.loc[pick(df, 0.04), "weight_kg"] = np.nan

    # Impossible / future DOBs
    df.loc[pick(df, 0.005), "date_of_birth"] = "2099-01-01"

    # Duplicate patients with slightly different details
    dupes = df.sample(frac=0.04).copy()
    dupes["patient_id"] = dupes["patient_id"].str.replace("PT", "PT9", regex=False)
    dupes["first_name"] = dupes["first_name"].str.strip().str[:-1]  # typo
    df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1).reset_index(drop=True)


def mess_up_encounters(clean: pd.DataFrame, valid_patient_ids) -> pd.DataFrame:
    df = clean.copy()

    # Visit dates in mixed formats
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%m/%d/%y"]
    df["visit_date"] = df["visit_date"].apply(lambda d: d.strftime(random.choice(formats)))
    df.loc[pick(df, 0.01), "visit_date"] = "2099-12-31"  # future date
    df.loc[pick(df, 0.01), "visit_date"] = np.nan

    # Diagnosis codes with formatting problems
    def scramble_code(c):
        return random.choice([c, c.replace(".", ""), c.lower(), f" {c}", c.replace(".", "") + " "])
    idx = pick(df, 0.20)
    df.loc[idx, "diagnosis_code"] = df.loc[idx, "diagnosis_code"].apply(scramble_code)
    df.loc[pick(df, 0.03), "diagnosis_code"] = np.nan
    df.loc[pick(df, 0.01), "diagnosis_code"] = "UNKNOWN"

    # Visit type variants
    variants = {
        "Office Visit": ["Office Visit", "office visit", "Office", "OV"],
        "Telehealth": ["Telehealth", "telehealth", "Tele", "Virtual"],
        "Annual Physical": ["Annual Physical", "Annual", "Physical", "annual physical"],
    }
    df["visit_type"] = df["visit_type"].apply(
        lambda v: random.choice(variants[v]) if v in variants else v
    )

    # Vitals: outliers, swapped BP, missing
    df.loc[pick(df, 0.01), "systolic_bp"] = 0
    df.loc[pick(df, 0.01), "heart_rate"] = 720   # typo
    idx = pick(df, 0.01)
    df.loc[idx, ["systolic_bp", "diastolic_bp"]] = df.loc[idx, ["diastolic_bp", "systolic_bp"]].values
    for col in ["systolic_bp", "diastolic_bp", "heart_rate"]:
        df.loc[pick(df, 0.05), col] = np.nan

    # Charges: negatives, text formatting, missing
    df["charge_amount"] = df["charge_amount"].astype(object)
    idx = pick(df, 0.10)
    df.loc[idx, "charge_amount"] = df.loc[idx, "charge_amount"].apply(lambda x: f"${x:,.2f}")
    df.loc[pick(df, 0.01), "charge_amount"] = -50.0
    df.loc[pick(df, 0.03), "charge_amount"] = np.nan

    # no_show recorded inconsistently
    df["no_show"] = df["no_show"].apply(
        lambda b: random.choice(["Y", "Yes", "TRUE", "1"] if b else ["N", "No", "FALSE", "0"])
    )
    df.loc[pick(df, 0.03), "no_show"] = np.nan

    # Orphan encounters (patient_id doesn't exist)
    orphans = df.sample(frac=0.01).index
    df.loc[orphans, "patient_id"] = [f"PT{random.randint(800000, 899999)}" for _ in orphans]

    # Exact duplicate rows
    df = pd.concat([df, df.sample(frac=0.03)], ignore_index=True)

    return df.sample(frac=1).reset_index(drop=True)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic messy healthcare data")
    parser.add_argument("--patients", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    fake = Faker("en_US")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    patients_clean = build_clean_patients(args.patients, fake)
    encounters_clean = build_clean_encounters(patients_clean, fake)

    patients_raw = mess_up_patients(patients_clean)
    encounters_raw = mess_up_encounters(encounters_clean, patients_clean["patient_id"])

    patients_clean.to_csv(out / "patients_clean.csv", index=False)
    encounters_clean.to_csv(out / "encounters_clean.csv", index=False)
    patients_raw.to_csv(out / "patients_raw.csv", index=False)
    encounters_raw.to_csv(out / "encounters_raw.csv", index=False)

    print(f"patients_raw.csv:   {len(patients_raw):>6} rows (clean key: {len(patients_clean)})")
    print(f"encounters_raw.csv: {len(encounters_raw):>6} rows (clean key: {len(encounters_clean)})")
    print(f"Files written to: {out.resolve()}")
    print("All data is synthetic. No real patient information is included.")


if __name__ == "__main__":
    main()
