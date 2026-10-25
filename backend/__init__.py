"""
Backend package — FastAPI application entry point.
---------------------------------------------------
The main FastAPI app lives in the project root (main.py) and is
started with:  uvicorn main:app --reload

This directory exists to satisfy the required project structure:
  data/         → dataset (air_quality.csv)
  model/        → trained ML model (aqi_model.pkl)
  ai-service/   → Gemini LLM integration
  backend/      → FastAPI backend reference
"""
