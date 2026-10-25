# Real-Time AQI Prediction and Context-Aware Health Advisory

## Project Overview
This project predicts the Air Quality Index (AQI) using machine learning and generates context-aware health advisories using a Large Language Model (LLM).

The system uses air-quality parameters to estimate AQI and provides health guidance based on the predicted air quality.

## Objectives
- Predict AQI using a trained machine learning model.
- Process air-quality parameters.
- Generate health advisories using an LLM.
- Provide a simple web interface for users to view predictions and advice.

## Technology Stack
- **Backend:** Python, FastAPI
- **Machine Learning:** Random Forest
- **LLM Integration:** Google Gemini API
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** CSV air-quality dataset
- **Model Storage:** Pickle (`.pkl`)

## Project Structure
- `ai-service/` — LLM-based health advisory logic
- `backend/` — Backend package
- `data/` — Air-quality dataset
- `model/` — Trained machine learning model
- `static/` — Frontend files
- `main.py` — Application entry point
- `train_model.py` — Model training script
- `requirements.txt` — Python dependencies
- `.env.example` — Example environment configuration

## Prerequisites
- Python 3.10 or a compatible version supported by the dependencies
- A Google Gemini API key for LLM features

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/prasanna-88/real-time-aqi-prediction.git
   cd real-time-aqi-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a local `.env` file using `.env.example` as a reference. Add your API key locally.

5. Start the application using the command appropriate for the entry point configured in `main.py`. For a FastAPI application, this may be:

   ```bash
   uvicorn main:app --reload
   ```

## Security
- Never upload `.env` or real API keys to GitHub.
- Keep API keys private and configure them locally.
- Do not commit virtual environments or Python cache files.

## Disclaimer
AQI predictions and generated health advisories are informational only and should not replace official air-quality reports or professional medical advice.
