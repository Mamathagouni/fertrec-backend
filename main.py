from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import joblib
import torch
import torch.nn as nn
import tensorflow as tf
import io
from PIL import Image
import uvicorn
import os
import traceback
import requests
import json
from datetime import datetime

# ==========================================================
# 0) ENV / CPU SETTINGS
# ==========================================================
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
torch.set_num_threads(4)

# ==========================================================
# 0.1) GEMINI REST CONFIG
# ==========================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = "gemini-2.5-flash"
gemini_enabled = bool(GEMINI_API_KEY)

# ==========================================================
# 1) FASTAPI APP
# ==========================================================
app = FastAPI(title="Smart Agriculture Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 2) PATHS
# ==========================================================
MODEL_DIR = "models"
ENC_DIR = "encoders"
IOT_STORE_FILE = "iot_data_store.json"

CROP_MODEL_PATH = os.path.join(MODEL_DIR, "rf_crop_opt_model.pkl")
LE_CROP_TARGET_PATH = os.path.join(MODEL_DIR, "le_crop_target.pkl")

SAINT_CHEM_PATH = os.path.join(MODEL_DIR, "saint_chemical_model.pt")
SAINT_ORG_PATH = os.path.join(MODEL_DIR, "saint_organic_model.pt")

SOIL_ENC_CHEM = os.path.join(ENC_DIR, "le_soil_chem.pkl")
CROP_ENC_CHEM = os.path.join(ENC_DIR, "le_crop_chem.pkl")
FERT_ENC_CHEM = os.path.join(ENC_DIR, "le_fert_chem.pkl")

SOIL_ENC_ORG = os.path.join(ENC_DIR, "le_soil_org.pkl")
CROP_ENC_ORG = os.path.join(ENC_DIR, "le_crop_org.pkl")
FERT_ENC_ORG = os.path.join(ENC_DIR, "le_fert_org.pkl")

DISEASE_SAVEDMODEL_DIR = os.path.join(MODEL_DIR, "tomato_disease_savedmodel")
DISEASE_H5_PATH = os.path.join(MODEL_DIR, "tomato_disease_mobilenetv2_abc.h5")

# ==========================================================
# 3) LOAD CROP MODEL
# ==========================================================
print("🔄 Loading crop model...")
crop_model = joblib.load(CROP_MODEL_PATH)
try:
    le_crop_target = joblib.load(LE_CROP_TARGET_PATH)
except Exception:
    le_crop_target = None

# ==========================================================
# 4) LOAD ENCODERS
# ==========================================================
print("🔄 Loading encoders...")
soil_encoder_chem = joblib.load(SOIL_ENC_CHEM)
crop_encoder_chem = joblib.load(CROP_ENC_CHEM)
fert_encoder_chem = joblib.load(FERT_ENC_CHEM)

soil_encoder_org = joblib.load(SOIL_ENC_ORG)
crop_encoder_org = joblib.load(CROP_ENC_ORG)
fert_encoder_org = joblib.load(FERT_ENC_ORG)

# ==========================================================
# 4.1) NORMALIZATION HELPERS
# ==========================================================
def _canon(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

def normalize_to_encoder(le, value: str) -> str:
    if value is None:
        return value
    v = str(value).strip()
    if not v:
        return v
    classes = list(map(str, le.classes_))
    cmap = {_canon(c): c for c in classes}
    return cmap.get(_canon(v), v)

def normalize_crop_for_saint(crop_value: str) -> str:
    if crop_value is None:
        return crop_value
    key = _canon(crop_value)

    veg_aliases = {
        "tomato", "brinjal", "eggplant", "chilli", "chili", "capsicum",
        "okra", "bhendi", "potato", "onion", "cabbage", "cauliflower",
        "beans", "carrot", "radish", "vegetable", "vegetables"
    }
    if key in veg_aliases:
        return "Vegetables"

    groups = ["Cotton", "Maize", "Pulses", "Rice", "Sugarcane", "Vegetables", "Wheat"]
    gmap = {_canon(g): g for g in groups}
    return gmap.get(key, str(crop_value).strip())

def _safe_le_transform(le, value: str, field_name: str) -> int:
    try:
        return int(le.transform([value])[0])
    except Exception:
        classes = list(map(str, le.classes_))
        raise ValueError(
            f"Unknown {field_name}='{value}'. Allowed values: {classes[:50]}{'...' if len(classes)>50 else ''}"
        )

# ==========================================================
# 5) SAINT MODEL
# ==========================================================
device = torch.device("cpu")

class SAINTCheckpointModel(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        depth: int,
        heads: int,
        ff_dim: int,
        n_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_tokens)])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.cls = nn.Sequential(
            nn.Linear(n_tokens * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        toks = []
        for i, layer in enumerate(self.embeds):
            fi = x[:, i].view(bsz, 1)
            toks.append(layer(fi))
        toks = torch.stack(toks, dim=1)
        z = self.transformer(toks)
        return self.cls(z.reshape(bsz, -1))

def _infer_n_tokens(sd: Dict[str, torch.Tensor]) -> int:
    idxs = []
    for k in sd.keys():
        if k.startswith("embeds.") and k.endswith(".weight"):
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    if not idxs:
        raise RuntimeError("SAINT checkpoint does not contain embeds.*.weight keys.")
    return max(idxs) + 1

def _infer_d_model(sd: Dict[str, torch.Tensor]) -> int:
    w = sd.get("embeds.0.weight", None)
    if w is None:
        for k, v in sd.items():
            if k.startswith("embeds.") and k.endswith(".weight"):
                w = v
                break
    if w is None:
        raise RuntimeError("Cannot infer d_model from checkpoint.")
    return int(w.shape[0])

def _infer_depth(sd: Dict[str, torch.Tensor]) -> int:
    layers = set()
    for k in sd.keys():
        if k.startswith("transformer.layers."):
            try:
                layers.add(int(k.split(".")[2]))
            except Exception:
                pass
    if not layers:
        raise RuntimeError("Cannot infer transformer depth.")
    return max(layers) + 1

def _infer_ff_dim(sd: Dict[str, torch.Tensor]) -> int:
    w = sd.get("transformer.layers.0.linear1.weight", None)
    if w is None:
        for k, v in sd.items():
            if "transformer.layers." in k and k.endswith(".linear1.weight"):
                w = v
                break
    if w is None:
        raise RuntimeError("Cannot infer ff_dim.")
    return int(w.shape[0])

def _infer_heads(d_model: int) -> int:
    return 4 if d_model % 4 == 0 else 1

def build_and_load_saint(state_path: str, is_chem: bool) -> nn.Module:
    n_classes = len(fert_encoder_chem.classes_) if is_chem else len(fert_encoder_org.classes_)
    sd = torch.load(state_path, map_location=device)

    n_tokens = _infer_n_tokens(sd)
    d_model = _infer_d_model(sd)
    depth = _infer_depth(sd)
    ff_dim = _infer_ff_dim(sd)
    heads = _infer_heads(d_model)

    model = SAINTCheckpointModel(
        n_tokens=n_tokens,
        d_model=d_model,
        depth=depth,
        heads=heads,
        ff_dim=ff_dim,
        n_classes=n_classes,
        dropout=0.1
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

print("🔄 Loading SAINT models...")
saint_chem = build_and_load_saint(SAINT_CHEM_PATH, is_chem=True)
saint_org = build_and_load_saint(SAINT_ORG_PATH, is_chem=False)
print("✅ SAINT models ready")

# ==========================================================
# 6) DISEASE MODEL
# ==========================================================
TOMATO_CLASSES = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_healthy",
]

_disease_model = None
_disease_model_type = None  # keras_h5 / keras_savedmodel / savedmodel_signature

def load_disease_model():
    global _disease_model, _disease_model_type

    if _disease_model is not None:
        return _disease_model

    if os.path.isfile(DISEASE_H5_PATH):
        try:
            model = tf.keras.models.load_model(DISEASE_H5_PATH, compile=False)
            _disease_model = model
            _disease_model_type = "keras_h5"
            print(f"✅ Disease H5 model loaded: {DISEASE_H5_PATH}")
            return _disease_model
        except Exception as e:
            print("⚠️ Failed to load H5 disease model:", e)

    if os.path.isdir(DISEASE_SAVEDMODEL_DIR):
        try:
            model = tf.keras.models.load_model(DISEASE_SAVEDMODEL_DIR, compile=False)
            if not hasattr(model, "predict"):
                raise RuntimeError("Loaded object does not support predict(); will try savedmodel_signature.")
            _disease_model = model
            _disease_model_type = "keras_savedmodel"
            print(f"✅ Disease SavedModel loaded via keras: {DISEASE_SAVEDMODEL_DIR}")
            return _disease_model
        except Exception as e:
            print("⚠️ Keras SavedModel load failed or unsupported:", e)

        try:
            sm = tf.saved_model.load(DISEASE_SAVEDMODEL_DIR)
            fn = sm.signatures.get("serving_default", None)
            if fn is None:
                raise RuntimeError("SavedModel has no 'serving_default' signature.")
            _disease_model = fn
            _disease_model_type = "savedmodel_signature"
            print(f"✅ Disease SavedModel loaded via tf.saved_model.load: {DISEASE_SAVEDMODEL_DIR}")
            return _disease_model
        except Exception as e:
            print("⚠️ tf.saved_model.load failed:", e)

    raise RuntimeError(
        f"No usable disease model found. Checked H5: {DISEASE_H5_PATH} and SavedModel: {DISEASE_SAVEDMODEL_DIR}"
    )

def preprocess_leaf(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)

def disease_predict(arr: np.ndarray) -> np.ndarray:
    model = load_disease_model()

    if _disease_model_type in ["keras_h5", "keras_savedmodel"]:
        if not hasattr(model, "predict"):
            raise RuntimeError("Disease model loaded as keras_savedmodel but does not support predict().")
        preds = model.predict(arr, verbose=0)
        return np.array(preds)

    if _disease_model_type == "savedmodel_signature":
        x = tf.convert_to_tensor(arr, dtype=tf.float32)
        out = model(x)
        if isinstance(out, dict):
            key = list(out.keys())[0]
            preds = out[key].numpy()
        else:
            preds = out.numpy()
        return np.array(preds)

    raise RuntimeError("Unknown disease model type")

# ==========================================================
# 7) FALLBACK LLM (Cloud-safe version)
# ==========================================================
print("🔄 Skipping local FLAN loading for cloud deployment...")
llm_pipeline = None
LLM_NAME_USED = "gemini_rest" if gemini_enabled else "rule_fallback"

def _normalize_question(q: str) -> str:
    return " ".join(q.strip().lower().replace("?", "").replace(".", "").replace("!", "").split())

def build_base_prompt(language: str, context: Dict[str, Any]) -> str:
    if language == "te":
        text = (
            "మీరు భారతదేశ రైతులకు సహాయం చేసే వ్యవసాయ నిపుణులు. "
            "రైతుకు సులభంగా అర్థమయ్యే తెలుగులో మాత్రమే సమాధానం ఇవ్వండి. "
            "ప్రశ్నను మళ్లీ రాయకండి. 3-5 చిన్న వాక్యాల్లో సమాధానం ఇవ్వండి.\n\n"
        )
    else:
        text = (
            "You are an agricultural expert for Indian farmers. "
            "Answer in simple English. Do not repeat the question. "
            "Write 3-5 short practical sentences.\n\n"
        )

    if context:
        ctx_text = "; ".join(f"{k}={v}" for k, v in context.items() if v is not None)
        text += f"Context: {ctx_text}\n\n"

    text += "Start with direct advice, then give practical tips."
    return text

def run_flan(question: str, language: str, context: Dict[str, Any]) -> str:
    return ""

def run_gemini(question: str, language: str, context: Dict[str, Any]) -> str:
    if not gemini_enabled:
        return ""

    base_prompt = build_base_prompt(language, context)
    full_prompt = (
        f"{base_prompt}\n\n"
        f"Farmer question: {question}\n\n"
        "Give a helpful answer. Do not repeat the question.\nAnswer:\n"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}]
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        return text
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        return ""

def run_llm(question: str, language: str, context: Dict[str, Any]) -> str:
    global LLM_NAME_USED

    out = run_gemini(question, language, context)
    if len(out) >= 20:
        LLM_NAME_USED = "gemini_rest"
        return out

    out = run_flan(question, language, context)
    if len(out) >= 20:
        LLM_NAME_USED = "flan_disabled"
        return out

    LLM_NAME_USED = "rule_fallback"
    if language == "te":
        return (
            "స్థానిక వ్యవసాయ అధికారుల లేదా మట్టి పరీక్ష సూచనలను పాటించండి. "
            "ఎరువులు అధికంగా వేయకండి. నీటిపారుదల సరిగ్గా నిర్వహించండి."
        )
    return (
        "Follow local agriculture officer or soil test recommendation. "
        "Avoid overuse of fertilizers and manage irrigation properly."
    )

# ==========================================================
# 8) STORAGE
# ==========================================================
farmer_decisions: List[Dict[str, Any]] = []

def load_iot_store() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(IOT_STORE_FILE):
        try:
            with open(IOT_STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_iot_store(data: Dict[str, Dict[str, Any]]) -> None:
    with open(IOT_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

last_iot_data: Dict[str, Dict[str, Any]] = load_iot_store()

def _has_iot(device_id: Optional[str] = None) -> bool:
    if device_id:
        item = last_iot_data.get(device_id)
        return item is not None
    return len(last_iot_data) > 0

# ==========================================================
# 9) DEFAULT Q&A
# ==========================================================
DEFAULT_QA = [
    {
        "id": "urea_rice",
        "question_en": "is urea suitable for rice?",
        "answer_en": (
            "Yes. Urea is commonly used as a nitrogen fertilizer for rice in chemical farming. "
            "Apply it in 2-3 split doses and irrigate after application to reduce losses."
        ),
        "question_te": "యూరియా వరి పంటకు సరిపోతుందా?",
        "answer_te": (
            "అవును. యూరియా వరి పంటకు సాధారణంగా ఉపయోగించే నత్రజని ఎరువు. "
            "2-3 విడతలుగా వేయండి మరియు తర్వాత నీరు ఇవ్వండి."
        ),
    },
    {
        "id": "mop_cotton",
        "question_en": "is mop suitable for cotton on black soil?",
        "answer_en": (
            "Yes. MOP can be suitable for cotton on black soil when potassium support is needed. "
            "Apply only as per recommended dose and avoid excess use."
        ),
        "question_te": "నల్ల నేలలో పత్తికి MOP సరిపోతుందా?",
        "answer_te": (
            "అవును. పొటాషియం అవసరం ఉన్నప్పుడు నల్ల నేలలో పత్తి పంటకు MOP ఉపయోగకరంగా ఉండొచ్చు. "
            "సిఫార్సు చేసిన మోతాదులో మాత్రమే వేయండి."
        ),
    },
]

DEFAULT_QA_MAP: Dict[str, Dict[str, str]] = {}
for qa in DEFAULT_QA:
    DEFAULT_QA_MAP[_normalize_question(qa["question_en"])] = qa
    DEFAULT_QA_MAP[_normalize_question(qa["question_te"])] = qa
    if qa["id"] == "urea_rice":
        DEFAULT_QA_MAP[_normalize_question("urea for rice")] = qa
        DEFAULT_QA_MAP[_normalize_question("urea for paddy")] = qa

# ==========================================================
# 10) REQUEST SCHEMAS
# ==========================================================
class IoTPayload(BaseModel):
    device_id: str
    Temperature: float
    Humidity: float
    Moisture: float

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    pH: float
    Rainfall: float
    explain: bool = True

class FertilizerRequest(BaseModel):
    farmer_type: str
    soil_type: str
    crop_type: str
    N: float
    P: float
    K: float
    Temperature: Optional[float] = None
    Humidity: Optional[float] = None
    Moisture: Optional[float] = None
    explain: bool = True
    mode: Optional[str] = "AUTO"
    top_k: Optional[int] = 3
    language: Optional[str] = "en"
    device_id: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    language: str
    context: Optional[Dict[str, Any]] = None

class DecisionSaveRequest(BaseModel):
    farmer_name: Optional[str] = None
    farmer_type: Optional[str] = None
    selected_fertilizer: str
    soil_type: Optional[str] = None
    crop_type: Optional[str] = None
    mode_used: Optional[str] = None
    confidence: Optional[float] = None
    decision: str = "APPLY"
    notes: Optional[str] = None

# ==========================================================
# 11) DSS HELPERS
# ==========================================================
def compute_irrigation_advice(moisture: Optional[float]) -> Dict[str, str]:
    if moisture is None:
        return {
            "status": "UNKNOWN",
            "advice_en": "Moisture data not available. Enter moisture manually or check sensor.",
            "advice_te": "తేమ డేటా లేదు. మాన్యువల్‌గా విలువ ఇవ్వండి లేదా సెన్సార్ చెక్ చేయండి.",
        }

    m = float(moisture)
    if m < 25:
        return {
            "status": "IRRIGATE_NOW",
            "advice_en": "Moisture is low (<25%). Give light irrigation now and recheck moisture.",
            "advice_te": "తేమ తక్కువ (<25%). వెంటనే తేలికైన నీటిపారుదల ఇవ్వండి.",
        }
    if m > 70:
        return {
            "status": "AVOID_IRRIGATION",
            "advice_en": "Moisture is high (>70%). Avoid irrigation now to prevent waterlogging.",
            "advice_te": "తేమ ఎక్కువ (>70%). ఇప్పుడు నీటిపారుదల ఇవ్వకండి.",
        }
    return {
        "status": "OK",
        "advice_en": "Moisture is normal (25-70%). Irrigation not required now.",
        "advice_te": "తేమ సాధారణం (25-70%). ప్రస్తుతం నీటిపారుదల అవసరం లేదు.",
    }

def nutrient_status_simple(n_val: float, p_val: float, k_val: float) -> Dict[str, str]:
    def lvl(x):
        if x < 40:
            return "LOW"
        if x < 80:
            return "MEDIUM"
        return "HIGH"
    return {"N": lvl(n_val), "P": lvl(p_val), "K": lvl(k_val)}

def dosage_plan_basic(farmer_type: str) -> Dict[str, str]:
    if farmer_type.lower().strip() == "organic":
        return {
            "en": "Organic plan: apply compost/FYM as basal and use bio-fertilizers as per label.",
            "te": "ఆర్గానిక్ ప్లాన్: బేసల్‌లో కంపోస్టు/FYM వేయండి, బయో-ఎరువులు లేబుల్ ప్రకారం వాడండి.",
        }
    return {
        "en": "Chemical plan: apply in split doses and irrigate after urea-type fertilizers. Avoid overuse.",
        "te": "కెమికల్ ప్లాన్: విడతలుగా వేయండి, యూరియా వంటి ఎరువుల తర్వాత నీరు ఇవ్వండి. అధిక మోతాదు వద్దు.",
    }

def build_data_quality(mode_used: str, temp_val, hum_val, moist_val) -> Dict[str, Any]:
    missing = []
    if temp_val is None:
        missing.append("Temperature")
    if hum_val is None:
        missing.append("Humidity")
    if moist_val is None:
        missing.append("Moisture")

    if len(missing) == 0:
        level = "LOW"
    elif len(missing) == 1:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "mode_used": mode_used,
        "missing_fields": missing,
        "uncertainty_level": level,
        "message": "All input values available." if not missing else f"Missing: {', '.join(missing)}",
    }

def compute_uncertainty_from_probs(top1: float, top2: float) -> Dict[str, Any]:
    top1 = float(top1)
    top2 = float(top2)
    gap = float(top1 - top2)

    if top1 >= 0.70 and gap >= 0.20:
        level = "LOW"
        msg = "Model confidence is high. Alternatives are clearly lower, so this recommendation is stable."
    elif top1 < 0.50 or gap < 0.10:
        level = "HIGH"
        msg = "Model confidence is low. Alternatives are close, so recheck inputs for best decision."
    else:
        level = "MEDIUM"
        msg = "Model confidence is moderate. Alternatives are somewhat close, so verify inputs for best decision."

    note = "Final decision is yours. If uncertainty is HIGH, recheck SoilType/CropType and confirm NPK values (or retest soil)."
    return {
        "level": level,
        "confidence": round(top1 * 100.0, 2),
        "margin": round(gap * 100.0, 2),
        "message": msg,
        "note": note,
    }

def recommended_action_text(farmer_type: str) -> str:
    if farmer_type.strip().lower() == "organic":
        return "Apply as per recommended dose. Prefer basal application of compost/FYM and maintain moisture for nutrient release."
    return "Apply as per recommended dose. If possible, apply in split doses for better uptake."

def disease_recommendation_text(disease_name: str) -> Dict[str, str]:
    name = (disease_name or "").lower()

    if "early_blight" in name:
        return {
            "en": "Remove infected leaves, avoid overhead irrigation, and apply a suitable fungicide if disease spreads.",
            "te": "బాధిత ఆకులను తొలగించండి, పై నుంచి నీరు పోయడం తగ్గించండి, వ్యాధి ఎక్కువైతే సరైన ఫంగిసైడ్ వాడండి.",
        }
    if "late_blight" in name:
        return {
            "en": "Remove infected leaves immediately, improve field aeration, and apply a recommended fungicide early.",
            "te": "బాధిత ఆకులను వెంటనే తొలగించండి, గాలి ప్రసరణ మెరుగుపరచండి, ప్రారంభ దశలో సరైన ఫంగిసైడ్ వాడండి.",
        }
    if "leaf_mold" in name:
        return {
            "en": "Reduce humidity, improve ventilation, and remove infected plant parts.",
            "te": "ఆర్ద్రత తగ్గించండి, గాలి ప్రసరణ మెరుగుపరచండి, బాధిత భాగాలను తొలగించండి.",
        }
    if "septoria" in name:
        return {
            "en": "Remove infected leaves, avoid water splash on leaves, and keep field sanitation clean.",
            "te": "బాధిత ఆకులను తొలగించండి, ఆకులపై నీరు చిమ్మడం తగ్గించండి, పొలాన్ని శుభ్రంగా ఉంచండి.",
        }
    if "spider_mites" in name:
        return {
            "en": "Spray water mist carefully under leaves, remove badly affected leaves, and use recommended mite control if needed.",
            "te": "ఆకుల క్రింద తేలికగా నీటి మిస్ట్ ఇవ్వండి, ఎక్కువగా దెబ్బతిన్న ఆకులను తొలగించండి, అవసరమైతే సరైన మైట్ నియంత్రణ వాడండి.",
        }
    if "healthy" in name:
        return {
            "en": "Leaf appears healthy. Continue balanced fertilization, proper irrigation, and regular monitoring.",
            "te": "ఆకు ఆరోగ్యంగా ఉంది. సమతుల్య ఎరువులు, సరైన నీటిపారుదల, క్రమం తప్పని పర్యవేక్షణ కొనసాగించండి.",
        }

    return {
        "en": "Monitor the crop regularly and consult a local agriculture officer if symptoms spread.",
        "te": "పంటను క్రమం తప్పకుండా గమనించండి. లక్షణాలు పెరిగితే స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.",
    }

# ==========================================================
# 12) SAINT PREDICT WITH PROBS
# ==========================================================
def saint_predict_with_probs(
    farmer_type: str,
    soil_type: str,
    crop_type: str,
    n_val: float,
    p_val: float,
    k_val: float,
    temperature: Optional[float],
    humidity: Optional[float],
    moisture: Optional[float],
    top_k: int = 3
) -> Tuple[str, float, List[Dict[str, Any]]]:
    is_chem = farmer_type.strip().lower() == "chemical"
    model = saint_chem if is_chem else saint_org

    soil_le = soil_encoder_chem if is_chem else soil_encoder_org
    crop_le = crop_encoder_chem if is_chem else crop_encoder_org
    fert_le = fert_encoder_chem if is_chem else fert_encoder_org

    soil_type = normalize_to_encoder(soil_le, soil_type)
    crop_type_used = normalize_crop_for_saint(crop_type)
    crop_type_used = normalize_to_encoder(crop_le, crop_type_used)

    s_id = _safe_le_transform(soil_le, soil_type, "soil_type")
    c_id = _safe_le_transform(crop_le, crop_type_used, "crop_type")

    x = np.array([[
        float(s_id),
        float(c_id),
        float(n_val), float(p_val), float(k_val),
        float(temperature if temperature is not None else 0.0),
        float(humidity if humidity is not None else 0.0),
        float(moisture if moisture is not None else 0.0),
    ]], dtype=np.float32)

    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    k = max(2, int(top_k or 3))
    k = min(k, probs.shape[0])
    idxs = np.argsort(-probs)[:k]

    alternatives = []
    for i in idxs:
        alternatives.append({
            "name": str(fert_le.classes_[int(i)]),
            "probability": round(float(probs[int(i)]) * 100.0, 1)
        })

    best_name = alternatives[0]["name"]
    best_conf = float(alternatives[0]["probability"]) / 100.0
    return best_name, best_conf, alternatives

# ==========================================================
# 13) CROP ID → NAME
# ==========================================================
CROP_ID_TO_NAME = {
    0: "Apple", 1: "Banana", 2: "Blackgram", 3: "Chickpea", 4: "Coconut", 5: "Coffee",
    6: "Cotton", 7: "Grape", 8: "Jute", 9: "Kidneybeans", 10: "Lentil", 11: "Maize",
    12: "Mango", 13: "Mothbeans", 14: "Mungbean", 15: "Muskmelon", 16: "Orange",
    17: "Papaya", 18: "Pigeonpeas", 19: "Pomegranate", 20: "Rice", 21: "Watermelon",
}

# ==========================================================
# 14) MODE RESOLUTION
# ==========================================================
def _resolve_mode_values(req: FertilizerRequest):
    mode = (req.mode or "AUTO").strip().upper()

    iot_item = None
    if req.device_id:
        iot_item = last_iot_data.get(req.device_id)

    iot_t = iot_item.get("Temperature") if iot_item else None
    iot_h = iot_item.get("Humidity") if iot_item else None
    iot_m = iot_item.get("Moisture") if iot_item else None

    manual_t = req.Temperature
    manual_h = req.Humidity
    manual_m = req.Moisture

    if mode == "IOT":
        if not req.device_id:
            raise HTTPException(status_code=400, detail="IOT mode requires device_id.")
        if not _has_iot(req.device_id):
            raise HTTPException(
                status_code=400,
                detail=f"IoT mode selected but no IoT data found for device_id={req.device_id}"
            )
        return (
            float(iot_t) if iot_t is not None else None,
            float(iot_h) if iot_h is not None else None,
            float(iot_m) if iot_m is not None else None,
            "IOT"
        )

    if mode == "OFFLINE":
        if manual_t is None or manual_h is None or manual_m is None:
            raise HTTPException(
                status_code=400,
                detail="Offline mode selected but Temperature/Humidity/Moisture were not provided."
            )
        return float(manual_t), float(manual_h), float(manual_m), "OFFLINE"

    temp_val = float(manual_t) if manual_t is not None else (float(iot_t) if iot_t is not None else None)
    hum_val = float(manual_h) if manual_h is not None else (float(iot_h) if iot_h is not None else None)
    moist_val = float(manual_m) if manual_m is not None else (float(iot_m) if iot_m is not None else None)

    used_manual = any(v is not None for v in [manual_t, manual_h, manual_m])
    used_iot = any(v is not None for v in [iot_t, iot_h, iot_m])

    if used_manual:
        mode_used = "OFFLINE"
    elif used_iot:
        mode_used = "IOT"
    else:
        mode_used = "UNKNOWN"

    return temp_val, hum_val, moist_val, mode_used

# ==========================================================
# 15) ENDPOINTS
# ==========================================================
@app.on_event("startup")
def startup_checks():
    print("🚀 API startup checks...")
    print("MODEL_DIR:", os.path.abspath(MODEL_DIR))
    print("ENC_DIR:", os.path.abspath(ENC_DIR))
    print("Disease SavedModel path:", os.path.abspath(DISEASE_SAVEDMODEL_DIR))
    print("Disease H5 path:", os.path.abspath(DISEASE_H5_PATH))
    print("IoT store path:", os.path.abspath(IOT_STORE_FILE))
    print("Gemini enabled:", gemini_enabled)

@app.get("/")
def home():
    return {
        "status": "API Running",
        "gemini_enabled": gemini_enabled,
        "llm_name": LLM_NAME_USED,
        "supported_modes": ["IOT", "OFFLINE", "AUTO"],
        "message": "Use /predict_iot for sensor input or /fertilizer/recommend with mode=OFFLINE for manual mode."
    }

@app.get("/app/config")
def app_config():
    return {
        "supported_modes": [
            {"key": "IOT", "label": "IoT Mode"},
            {"key": "OFFLINE", "label": "Manual Mode"},
            {"key": "AUTO", "label": "Auto Select"},
        ],
        "supported_languages": [
            {"key": "en", "label": "English"},
            {"key": "te", "label": "Telugu"},
        ],
        "advanced_fields": [
            "confidence",
            "uncertainty",
            "alternatives",
            "recommended_action"
        ]
    }

@app.post("/predict_iot")
def receive_iot(data: IoTPayload):
    global last_iot_data

    entry = {
        "device_id": data.device_id,
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "Moisture": data.Moisture,
        "timestamp": datetime.now().isoformat()
    }

    last_iot_data[data.device_id] = entry
    save_iot_store(last_iot_data)

    return {
        "message": "IoT data stored successfully",
        "data": entry
    }

@app.get("/iot/all")
def get_all_iot():
    return {
        "count": len(last_iot_data),
        "items": last_iot_data
    }

@app.get("/iot/latest")
def get_latest_iot_summary():
    if not last_iot_data:
        mock = {
            "device_id": "demo_device",
            "Temperature": 28.5,
            "Humidity": 58.0,
            "Moisture": 36.0,
            "timestamp": datetime.now().isoformat()
        }
        advice = compute_irrigation_advice(mock["Moisture"])
        return {
            **mock,
            "is_mock": True,
            "irrigation_advice": advice
        }

    latest_item = max(
        last_iot_data.values(),
        key=lambda x: x.get("timestamp", "")
    )
    advice = compute_irrigation_advice(latest_item.get("Moisture"))
    return {
        **latest_item,
        "is_mock": False,
        "irrigation_advice": advice
    }

@app.get("/iot/latest/{device_id}")
def get_iot_by_device(device_id: str):
    item = last_iot_data.get(device_id)

    if not item:
        raise HTTPException(status_code=404, detail=f"No IoT data found for device_id={device_id}")

    advice = compute_irrigation_advice(item.get("Moisture"))
    return {
        **item,
        "irrigation_advice": advice
    }

@app.get("/ask/defaults")
def get_default_qa():
    latest_summary = None
    if last_iot_data:
        latest_summary = max(last_iot_data.values(), key=lambda x: x.get("timestamp", ""))

    return {
        "defaults": DEFAULT_QA,
        "iot_data": latest_summary
    }

@app.get("/fertilizer/options")
def fertilizer_options():
    return {
        "soil_chemical": list(map(str, soil_encoder_chem.classes_)),
        "crop_chemical": list(map(str, crop_encoder_chem.classes_)),
        "fert_chemical": list(map(str, fert_encoder_chem.classes_)),
        "soil_organic": list(map(str, soil_encoder_org.classes_)),
        "crop_organic": list(map(str, crop_encoder_org.classes_)),
        "fert_organic": list(map(str, fert_encoder_org.classes_)),
        "note": "Tomato and related vegetable crops are mapped to Vegetables automatically."
    }

@app.get("/disease/health")
def disease_health():
    try:
        load_disease_model()
        return {
            "ok": True,
            "model_type": _disease_model_type,
            "savedmodel_path": os.path.abspath(DISEASE_SAVEDMODEL_DIR),
            "h5_path": os.path.abspath(DISEASE_H5_PATH),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "savedmodel_path": os.path.abspath(DISEASE_SAVEDMODEL_DIR),
            "h5_path": os.path.abspath(DISEASE_H5_PATH),
        }

@app.post("/crop/recommend")
def crop_recommend(req: CropRequest):
    X = np.array([[float(req.N), float(req.P), float(req.K),
                   float(req.Temperature), float(req.Humidity),
                   float(req.pH), float(req.Rainfall)]], dtype=np.float32)

    pred_id = int(crop_model.predict(X)[0])
    crop_name = CROP_ID_TO_NAME.get(pred_id, f"Crop_{pred_id}")

    latest_summary = None
    if last_iot_data:
        latest_summary = max(last_iot_data.values(), key=lambda x: x.get("timestamp", ""))

    response: Dict[str, Any] = {
        "recommended_crop": crop_name,
        "crop_id": pred_id,
        "input": req.model_dump(),
        "iot_data": latest_summary,
        "llm_name": LLM_NAME_USED,
    }

    if req.explain:
        ctx = {
            "crop": crop_name,
            "Temperature": float(req.Temperature),
            "Humidity": float(req.Humidity),
            "pH": float(req.pH),
            "Rainfall": float(req.Rainfall),
        }
        response["llm_explanation_en"] = run_llm(
            f"Why is {crop_name} suitable? Give 3 tips.",
            "en",
            ctx
        )
        response["llm_explanation_te"] = run_llm(
            f"{crop_name} ఎందుకు అనుకూలం? 3 సూచనలు ఇవ్వండి.",
            "te",
            ctx
        )
        response["llm_name"] = LLM_NAME_USED

    return response

@app.post("/fertilizer/recommend")
def fert_recommend(req: FertilizerRequest):
    temp_val, hum_val, moist_val, mode_used = _resolve_mode_values(req)

    irrigation = compute_irrigation_advice(moist_val)
    nutri = nutrient_status_simple(req.N, req.P, req.K)
    plan = dosage_plan_basic(req.farmer_type)
    data_quality = build_data_quality(mode_used, temp_val, hum_val, moist_val)

    try:
        fert, best_conf, alternatives = saint_predict_with_probs(
            farmer_type=req.farmer_type,
            soil_type=req.soil_type,
            crop_type=req.crop_type,
            n_val=req.N,
            p_val=req.P,
            k_val=req.K,
            temperature=temp_val,
            humidity=hum_val,
            moisture=moist_val,
            top_k=int(req.top_k or 3)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    top1 = alternatives[0]["probability"] / 100.0
    top2 = alternatives[1]["probability"] / 100.0 if len(alternatives) > 1 else 0.0
    uncertainty = compute_uncertainty_from_probs(top1, top2)
    recommended_action = recommended_action_text(req.farmer_type)

    language = (req.language or "en").lower()
    selected_iot = last_iot_data.get(req.device_id) if req.device_id else None

    base_response: Dict[str, Any] = {
        "recommended_fertilizer": fert,
        "confidence": round(top1 * 100.0, 1),
        "model_used": "SAINT",
        "mode_used": mode_used,
        "llm_name": LLM_NAME_USED,
        "uncertainty": uncertainty,
        "alternatives": alternatives,
        "recommended_action": recommended_action,
        "input": {
            "farmer_type": req.farmer_type,
            "soil_type": req.soil_type,
            "crop_type": req.crop_type,
            "crop_type_used_by_model": normalize_crop_for_saint(req.crop_type),
            "N": float(req.N),
            "P": float(req.P),
            "K": float(req.K),
            "Temperature": temp_val,
            "Humidity": hum_val,
            "Moisture": moist_val,
            "device_id": req.device_id,
        },
        "decision_support": {
            "irrigation_advice": irrigation,
            "nutrient_status": nutri,
            "dosage_plan": plan,
        },
        "data_quality": data_quality,
        "iot_data": selected_iot,
    }

    if req.explain:
        ctx = {
            "recommended_fertilizer": fert,
            "confidence_percent": base_response["confidence"],
            "uncertainty_level": uncertainty["level"],
            "soil_type": req.soil_type,
            "crop_type": req.crop_type,
            "N": float(req.N),
            "P": float(req.P),
            "K": float(req.K),
            "Temperature": temp_val,
            "Humidity": hum_val,
            "Moisture": moist_val,
            "irrigation_status": irrigation.get("status"),
            "device_id": req.device_id,
        }

        q_en = (
            f"Fertilizer recommendation: {fert}. "
            f"Give a short explanation for {req.crop_type} in {req.soil_type} soil using NPK and moisture. "
            f"Then give 3 safe application tips. Mention uncertainty level {uncertainty['level']}."
        )
        q_te = (
            f"సిఫార్సు చేసిన ఎరువు: {fert}. "
            f"{req.soil_type} నేలలో {req.crop_type} పంటకు NPK మరియు తేమ ఆధారంగా చిన్న వివరణ ఇవ్వండి. "
            f"తర్వాత 3 సురక్షిత వినియోగ సూచనలు ఇవ్వండి. అనిశ్చితి స్థాయి {uncertainty['level']} అని పేర్కొనండి."
        )

        base_response["llm_explanation_en"] = run_llm(q_en, "en", ctx)
        base_response["llm_explanation_te"] = run_llm(q_te, "te", ctx)
        base_response["llm_name"] = LLM_NAME_USED
        base_response["explanation"] = (
            base_response["llm_explanation_te"]
            if language == "te"
            else base_response["llm_explanation_en"]
        )

    return base_response

@app.post("/decision/save")
def save_farmer_decision(req: DecisionSaveRequest):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "farmer_name": req.farmer_name,
        "farmer_type": req.farmer_type,
        "selected_fertilizer": req.selected_fertilizer,
        "soil_type": req.soil_type,
        "crop_type": req.crop_type,
        "mode_used": req.mode_used,
        "confidence": req.confidence,
        "decision": req.decision,
        "notes": req.notes,
    }
    farmer_decisions.append(entry)
    return {
        "message": "Farmer decision saved",
        "saved": entry
    }

@app.get("/decision/all")
def get_all_decisions():
    return {
        "count": len(farmer_decisions),
        "items": farmer_decisions
    }

@app.post("/disease/detect")
async def detect_leaf(
    file: UploadFile = File(...),
    explain: bool = Form(True)
):
    try:
        img_bytes = await file.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        arr = preprocess_leaf(img_bytes)
        preds = disease_predict(arr)

        if preds.ndim == 2:
            preds = preds[0]

        if len(preds) != len(TOMATO_CLASSES):
            raise RuntimeError(
                f"Disease model output mismatch. Expected {len(TOMATO_CLASSES)} classes, got {len(preds)}"
            )

        idx = int(np.argmax(preds))
        disease_name = str(TOMATO_CLASSES[idx])
        confidence = float(preds[idx])
        reco = disease_recommendation_text(disease_name)

        latest_summary = None
        if last_iot_data:
            latest_summary = max(last_iot_data.values(), key=lambda x: x.get("timestamp", ""))

        response: Dict[str, Any] = {
            "disease": disease_name,
            "confidence": round(confidence, 4),
            "recommendation_en": reco["en"],
            "recommendation_te": reco["te"],
            "all_scores": {
                TOMATO_CLASSES[i]: round(float(preds[i]), 4)
                for i in range(len(TOMATO_CLASSES))
            },
            "iot_data": latest_summary,
            "llm_name": LLM_NAME_USED,
        }

        if explain:
            ctx = {"disease": disease_name, "confidence": confidence}
            if latest_summary:
                ctx.update(latest_summary)

            q_en = f"Explain {disease_name} symptoms and 2-3 safe management steps for farmers."
            q_te = f"{disease_name} లక్షణాలు మరియు రైతులకు 2-3 సురక్షిత నిర్వహణ సూచనలు ఇవ్వండి."
            response["llm_explanation_en"] = run_llm(q_en, "en", ctx)
            response["llm_explanation_te"] = run_llm(q_te, "te", ctx)
            response["llm_name"] = LLM_NAME_USED

        return response

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Disease detection error: {str(e)}")

@app.post("/ask")
def ask_ai(req: AskRequest):
    latest_summary = None
    if last_iot_data:
        latest_summary = max(last_iot_data.values(), key=lambda x: x.get("timestamp", ""))

    base_ctx = {
        "Temperature": latest_summary.get("Temperature") if latest_summary else None,
        "Humidity": latest_summary.get("Humidity") if latest_summary else None,
        "Moisture": latest_summary.get("Moisture") if latest_summary else None,
        "device_id": latest_summary.get("device_id") if latest_summary else None,
    }
    context = {**base_ctx, **(req.context or {})}

    norm_q = _normalize_question(req.question)
    qa = DEFAULT_QA_MAP.get(norm_q)
    if qa is not None:
        answer = qa["answer_te"] if req.language == "te" else qa["answer_en"]
        return {
            "answer": answer,
            "language": req.language,
            "from_default": True,
            "default_id": qa["id"],
            "context_used": context,
            "iot_data": latest_summary,
            "llm_name": "default_rule",
        }

    answer = run_llm(req.question, req.language, context)
    return {
        "answer": answer,
        "language": req.language,
        "from_default": False,
        "context_used": context,
        "iot_data": latest_summary,
        "llm_name": LLM_NAME_USED,
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)