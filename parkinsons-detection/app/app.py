# app/app.py

"""
Flask Web App for Parkinson's Detection
Routes:
  GET  /          → Main UI
  POST /analyze/voice  → Analyze uploaded audio
  POST /analyze/touch  → Analyze tap data JSON
  GET  /health    → Health check
"""

import os
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Adjust path for imports when running from app/ directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.voice_analysis.voice_pipeline import predict_voice
from src.touch_analysis.touch_features import extract_all_touch_features
from src.touch_analysis.touch_model import predict_touch
from src.models.fusion_model import fuse_scores, format_report

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "flac"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/analyze/voice", methods=["POST"])
def analyze_voice():
    """
    Accepts multipart/form-data with 'audio' file.
    Returns JSON with PD probability and risk level.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Use: {ALLOWED_AUDIO_EXTENSIONS}"}), 400

    # Save temp file
    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_voice(tmp_path)
        return jsonify({
            "success": True,
            "voice_probability": result["probability"],
            "risk_level": result["risk_level"],
            "risk_message": result["risk_message"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/analyze/touch", methods=["POST"])
def analyze_touch():
    """
    Accepts JSON body:
    {
      "taps": [
        {"timestamp": 0.0, "x": 100, "y": 200, "pressure": 0.5, "duration": 0.1},
        ...
      ]
    }
    Returns JSON with touch PD probability.
    """
    data = request.get_json()
    if not data or "taps" not in data:
        return jsonify({"error": "Expected JSON with 'taps' array"}), 400

    taps = data["taps"]
    if len(taps) < 5:
        return jsonify({"error": "Need at least 5 taps for analysis"}), 400

    import pandas as pd
    df = pd.DataFrame(taps)

    try:
        features = extract_all_touch_features(df)
        result = predict_touch(features)
        return jsonify({
            "success": True,
            "touch_probability": result["touch_pd_probability"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/combined", methods=["POST"])
def analyze_combined():
    """
    Accepts multipart/form-data with optional 'audio' file
    and optional JSON 'touch_data' field.
    Returns fused risk assessment.
    """
    voice_prob = None
    touch_prob = None

    # Voice
    if "audio" in request.files:
        file = request.files["audio"]
        if file and allowed_file(file.filename):
            suffix = "." + file.filename.rsplit(".", 1)[1].lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                voice_result = predict_voice(tmp_path)
                voice_prob = voice_result["probability"]
            finally:
                os.unlink(tmp_path)

    # Touch
    touch_json = request.form.get("touch_data")
    if touch_json:
        try:
            import pandas as pd
            taps = json.loads(touch_json)
            df = pd.DataFrame(taps)
            features = extract_all_touch_features(df)
            touch_result = predict_touch(features)
            touch_prob = touch_result["touch_pd_probability"]
        except Exception:
            pass

    if voice_prob is None and touch_prob is None:
        return jsonify({"error": "No valid audio or touch data provided"}), 400

    fusion = fuse_scores(voice_prob, touch_prob)
    report = format_report(fusion)

    return jsonify({
        "success": True,
        **fusion,
        "report": report,
    })


if __name__ == "__main__":
    print("🧠 Parkinson's Detection System starting...")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
