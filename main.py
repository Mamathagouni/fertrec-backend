from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import os

app = FastAPI()

# -----------------------------
# Firebase initialization
# -----------------------------
firebase_ready = False
db = None

try:
    FIREBASE_KEY_PATH = "firebase_key.json"

    # Render secret files path
    if not os.path.exists(FIREBASE_KEY_PATH):
        FIREBASE_KEY_PATH = "/etc/secrets/firebase_key.json"

    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    firebase_ready = True
    print("Firebase connected")

except Exception as e:
    firebase_ready = False
    db = None
    print("Firebase not connected:", e)


# -----------------------------
# Data model
# -----------------------------
class IoTData(BaseModel):
    device_id: str
    Temperature: float
    Humidity: float
    Moisture: float


# -----------------------------
# Memory cache
# -----------------------------
last_iot_data = {}


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "API Running",
        "supported_modes": ["IOT", "OFFLINE", "AUTO"],
        "message": "Use /predict_iot for sensor input"
    }


# -----------------------------
# IoT receive endpoint
# -----------------------------
@app.post("/predict_iot")
def receive_iot(data: IoTData):

    entry = {
        "device_id": data.device_id,
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "Moisture": data.Moisture,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save locally
    last_iot_data[data.device_id] = entry

    # -----------------------------
    # Irrigation Advice
    # -----------------------------
    moisture = data.Moisture

    if moisture < 25:
        status = "LOW"
        advice_en = "Soil moisture low. Irrigation required."
        advice_te = "నేల తేమ తక్కువగా ఉంది. నీరు పెట్టాలి."
    elif moisture <= 70:
        status = "OK"
        advice_en = "Moisture is normal (25–70%). Irrigation not required now."
        advice_te = "తేమ సాధారణంగా ఉంది (25–70%). ప్రస్తుతం నీరు అవసరం లేదు."
    else:
        status = "HIGH"
        advice_en = "Soil moisture high. Do not irrigate."
        advice_te = "నేల తేమ ఎక్కువగా ఉంది. నీరు పెట్టకండి."

    irrigation = {
        "status": status,
        "advice_en": advice_en,
        "advice_te": advice_te
    }

    # -----------------------------
    # Save to Firestore
    # -----------------------------
    firestore_status = "not_connected"

    if firebase_ready and db is not None:
        try:
            db.collection("iot_sensor_data").add(entry)
            firestore_status = "saved"
        except Exception as e:
            firestore_status = str(e)

    return {
        "message": "IoT data stored successfully",
        "data": entry,
        "irrigation_advice": irrigation,
        "firestore_status": firestore_status
    }


# -----------------------------
# Latest IoT (local)
# -----------------------------
@app.get("/iot/latest")
def latest():

    if not last_iot_data:
        return {"message": "no data"}

    device_id = list(last_iot_data.keys())[-1]
    data = last_iot_data[device_id]

    return data


# -----------------------------
# All IoT (local)
# -----------------------------
@app.get("/iot/all")
def all_iot():

    return {
        "count": len(last_iot_data),
        "items": last_iot_data
    }


# -----------------------------
# Cloud latest (Firestore)
# -----------------------------
@app.get("/cloud/latest")
def cloud_latest():

    if not firebase_ready or db is None:
        return {"error": "firebase not connected"}

    try:
        docs = db.collection("iot_sensor_data") \
            .order_by("timestamp", direction=firestore.Query.DESCENDING) \
            .limit(1).stream()

        for d in docs:
            data = d.to_dict()
            data["doc_id"] = d.id
            return data

        return {"message": "no data"}

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Cloud all (Firestore)
# -----------------------------
@app.get("/cloud/all")
def cloud_all():

    if not firebase_ready or db is None:
        return {"error": "firebase not connected"}

    try:
        docs = db.collection("iot_sensor_data").stream()

        data = []
        for d in docs:
            item = d.to_dict()
            item["doc_id"] = d.id
            data.append(item)

        return {
            "count": len(data),
            "items": data
        }

    except Exception as e:
        return {"error": str(e)}
