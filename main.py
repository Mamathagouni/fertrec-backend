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
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_ready = True
    print("Firebase connected")
except Exception as e:
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
# Root
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "API Running",
        "supported_modes": ["IOT","OFFLINE","AUTO"],
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

    # save locally
    last_iot_data[data.device_id] = entry

    # irrigation advice
    moisture = data.Moisture

    if moisture < 25:
        status = "LOW"
        advice_en = "Soil moisture low. Irrigation required."
        advice_te = "నేల తేమ తక్కువగా ఉంది. నీరు పెట్టాలి."
    elif moisture <= 70:
        status = "OK"
        advice_en = "Moisture is normal (25-70%). Irrigation not required now."
        advice_te = "తేమ సాధారణంగా ఉంది (25-70%). ప్రస్తుతం నీరు అవసరం లేదు."
    else:
        status = "HIGH"
        advice_en = "Soil moisture high. Do not irrigate."
        advice_te = "నేల తేమ ఎక్కువగా ఉంది. నీరు పెట్టకండి."

    irrigation = {
        "status": status,
        "advice_en": advice_en,
        "advice_te": advice_te
    }

    # save to Firestore
    firestore_status = "not_connected"

    if firebase_ready:
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
# Latest IoT
# -----------------------------
@app.get("/iot/latest")
def latest():

    if not last_iot_data:
        return {"message": "no data"}

    device_id = list(last_iot_data.keys())[-1]
    data = last_iot_data[device_id]

    return data

# -----------------------------
# All IoT
# -----------------------------
@app.get("/iot/all")
def all_iot():

    return {
        "count": len(last_iot_data),
        "items": last_iot_data
    }

# -----------------------------
# Cloud latest
# -----------------------------
@app.get("/cloud/latest")
def cloud_latest():

    if not firebase_ready:
        return {"error": "firebase not connected"}

    docs = db.collection("iot_sensor_data") \
        .order_by("timestamp", direction=firestore.Query.DESCENDING) \
        .limit(1).stream()

    for d in docs:
        return d.to_dict()

    return {"message":"no data"}

# -----------------------------
# Cloud all
# -----------------------------
@app.get("/cloud/all")
def cloud_all():

    if not firebase_ready:
        return {"error": "firebase not connected"}

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
