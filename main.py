"""
FastAPI backend for AQI Prediction & Health Advisory
-----------------------------------------------------
Endpoints:
  POST /predict  – predict AQI, categorize, generate Gemini health advice, store in MongoDB
  GET  /         – serve the frontend
"""

import os
import sys
import datetime
import importlib.util

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# App & ML model
# ---------------------------------------------------------------------------
app = FastAPI(title="AQI Prediction & Health Advisory")

# Allow CORS so frontend can call backend from any origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "aqi_model.pkl")
try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"[OK] Random Forest model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. Run `python train_model.py` first."
    )

# ---------------------------------------------------------------------------
# AI Service (Gemini integration from ai-service/)
# ---------------------------------------------------------------------------
_AI_SERVICE_PATH = os.path.join(os.path.dirname(__file__), "ai-service", "gemini_advice.py")
try:
    _spec = importlib.util.spec_from_file_location("gemini_advice", _AI_SERVICE_PATH)
    gemini_advice = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(gemini_advice)
    print(f"[OK] AI service loaded from {_AI_SERVICE_PATH}")
except Exception as e:
    gemini_advice = None
    print(f"[WARN] AI service not available ({e}). Using fallback advice.")

# ---------------------------------------------------------------------------
# MongoDB (async via motor) — completely optional
# ---------------------------------------------------------------------------
mongo_client = None
db = None
collection = None

try:
    from motor.motor_asyncio import AsyncIOMotorClient

    MONGO_URI = os.getenv("MONGODB_URI", "")

    @app.on_event("startup")
    async def startup_db():
        global mongo_client, db, collection
        if not MONGO_URI:
            print("[INFO] MONGODB_URI not set. Predictions will NOT be persisted.")
            return
        try:
            mongo_client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            await mongo_client.admin.command("ping")
            db = mongo_client["aqi_app"]
            collection = db["predictions"]
            print("[OK] Connected to MongoDB")
        except Exception as e:
            print(f"[WARN] MongoDB not available ({e}). Predictions will NOT be persisted.")
            mongo_client = None
            db = None
            collection = None

    @app.on_event("shutdown")
    async def shutdown_db():
        if mongo_client:
            mongo_client.close()

except ImportError:
    print("[INFO] motor not installed. MongoDB persistence disabled.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    pm2_5: float = Field(..., ge=0, description="PM2.5 (µg/m³)")
    pm10: float = Field(..., ge=0, description="PM10 (µg/m³)")
    co: float = Field(..., ge=0, description="CO (mg/m³)")
    no2: float = Field(..., ge=0, description="NO₂ (µg/m³)")
    so2: float = Field(..., ge=0, description="SO₂ (µg/m³)")
    o3: float = Field(..., ge=0, description="O₃ (µg/m³)")
    temperature: float = Field(..., description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    # User details for personalised advice
    age: int | None = Field(None, ge=0, le=120, description="User age")
    health_condition: str = Field(
        "none",
        description="Health condition: none, asthma, heart_disease, diabetes, respiratory, other",
    )
    activity: str = Field(
        "moderate",
        description="Activity level: sedentary, light, moderate, vigorous, outdoor_worker",
    )


class PredictionResponse(BaseModel):
    aqi: float
    category: str
    color: str
    health_advice: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def categorize_aqi(aqi: float) -> tuple[str, str]:
    """Return (category_label, hex_color) for the given AQI value."""
    if aqi <= 50:
        return "Good", "#4caf50"
    elif aqi <= 100:
        return "Moderate", "#ff9800"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff5722"
    elif aqi <= 200:
        return "Unhealthy", "#e91e63"
    elif aqi <= 300:
        return "Very Unhealthy", "#9c27b0"
    else:
        return "Hazardous", "#b71c1c"


async def generate_health_advice(
    aqi: float, category: str, req: PredictionRequest
) -> str:
    """Use Gemini (via ai-service/) to produce personalised health advice."""
    # Try the AI service first
    if gemini_advice and gemini_advice.is_available():
        pollutants = {
            "pm2_5": req.pm2_5, "pm10": req.pm10, "co": req.co,
            "no2": req.no2, "so2": req.so2, "o3": req.o3,
            "temperature": req.temperature, "humidity": req.humidity,
        }
        user_info = {
            "age": req.age,
            "health_condition": req.health_condition,
            "activity": req.activity,
        }
        advice = await gemini_advice.get_health_advice(aqi, category, pollutants, user_info)
        if advice:
            return advice

    # Fallback when Gemini is unavailable
    return _fallback_advice(aqi, category, req)


def _fallback_advice(aqi: float, category: str, req: PredictionRequest | None = None) -> str:
    """Deterministic advice when Gemini is unavailable."""

    # Build personalised prefix if user details are provided
    personal_notes = []
    if req:
        if req.health_condition and req.health_condition != "none":
            cond = req.health_condition.replace("_", " ").title()
            personal_notes.append(f"• ⚕️ Since you have {cond}, take extra precautions.")
        if req.activity in ("vigorous", "outdoor_worker"):
            personal_notes.append("• 🏃 Your high activity level increases exposure — limit intensity if AQI is poor.")
        if req.age is not None and (req.age < 12 or req.age > 65):
            personal_notes.append("• 👤 Your age group is more sensitive to air pollution — be extra cautious.")

    personal_block = "\n".join(personal_notes)
    if personal_block:
        personal_block += "\n"

    if aqi <= 50:
        return (
            personal_block
            + "• Air quality is excellent – enjoy outdoor activities!\n"
            "• No special precautions needed.\n"
            "• Great day for exercise outdoors."
        )
    elif aqi <= 100:
        return (
            personal_block
            + "• Air quality is acceptable.\n"
            "• Sensitive individuals should consider limiting prolonged outdoor exertion.\n"
            "• Keep windows open for ventilation."
        )
    elif aqi <= 200:
        return (
            personal_block
            + "• Reduce prolonged outdoor exertion.\n"
            "• Sensitive groups (children, elderly, asthmatics) should stay indoors.\n"
            "• Use air purifiers if available.\n"
            "• Wear an N95 mask outdoors."
        )
    else:
        return (
            personal_block
            + "• AVOID all outdoor physical activities.\n"
            "• Keep all windows and doors closed.\n"
            "• Use air purifiers on high setting.\n"
            "• Wear N95/P100 mask if you must go outside.\n"
            "• Seek medical attention if you experience difficulty breathing."
        )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Predict AQI, categorize, generate advice, and store the query."""
    features = np.array(
        [[req.pm2_5, req.pm10, req.co, req.no2, req.so2, req.o3, req.temperature, req.humidity]]
    )

    aqi_value = float(rf_model.predict(features)[0])
    aqi_value = round(max(0, min(aqi_value, 500)), 1)

    category, color = categorize_aqi(aqi_value)
    advice = await generate_health_advice(aqi_value, category, req)

    # Persist to MongoDB (if available)
    if collection is not None:
        try:
            doc = {
                "timestamp": datetime.datetime.utcnow(),
                "inputs": req.model_dump(),
                "aqi": aqi_value,
                "category": category,
                "advice": advice,
            }
            await collection.insert_one(doc)
        except Exception as e:
            print(f"MongoDB write error: {e}")

    return PredictionResponse(
        aqi=aqi_value,
        category=category,
        color=color,
        health_advice=advice,
    )


# ---------------------------------------------------------------------------
# Static files & SPA fallback
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
