import os
import joblib

ENC_DIR = "encoders"

files_to_check = [
    "le_soil_chem.pkl",
    "le_crop_chem.pkl",
    "le_fert_chem.pkl",
    "scaler_chem.pkl",
    "le_soil_org.pkl",
    "le_crop_org.pkl",
    "le_fert_org.pkl",
    "scaler_org.pkl",
]

print("\n✅ Checking encoder/scaler files...\n")

for f in files_to_check:
    path = os.path.join(ENC_DIR, f)
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        continue

    obj = joblib.load(path)
    print(f"✅ Loaded: {f}  -->  type: {type(obj)}")

    # If LabelEncoder, show classes
    if hasattr(obj, "classes_"):
        print(f"   classes_ count = {len(obj.classes_)}")
        print(f"   first 10 classes = {list(obj.classes_)[:10]}")

    # If scaler, show shape
    if hasattr(obj, "mean_"):
        print(f"   scaler features = {len(obj.mean_)}")

    print("-" * 60)

print("\n✅ Done.\n")