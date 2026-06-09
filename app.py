"""
Flask Web Application for NBL MAX PROFIT SYSTEM v5.2
Provides UI for CSV upload, model prediction, and parameter configuration.
"""
import os
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session

# ---------------------------------------------------------------------------
# Import model utilities from train_test11
# ---------------------------------------------------------------------------
from train_test11 import engineer_features, make_target, NBLMaxProfitV5

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_DIR = Path("nbl_v5_output") / "model"

# ---------------------------------------------------------------------------
# Global model instance (loaded once at startup)
# ---------------------------------------------------------------------------
system = None


def load_model():
    """Load pre-trained model from disk."""
    global system
    try:
        system = NBLMaxProfitV5(fee_percent=0.003)
        system.load(str(MODEL_DIR))
        print("[OK] Model loaded from", MODEL_DIR)
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        system = None


# ---------------------------------------------------------------------------
# Helper: run prediction on a CSV dataframe
# ---------------------------------------------------------------------------
def predict_from_df(df, buy_thresh, sell_thresh, target_horizon, target_threshold, fee_pct):
    """
    Run prediction on every valid row of uploaded CSV.
    Returns a list of per-row prediction dicts and a summary dict.
    """
    if system is None or system.model is None:
        return None, "Model not loaded. Please train the model first."

    # Prepare data
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    drop_cols = ["Symbol", "next_day_return", "direction"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Engineer features
    engineered = engineer_features(df)

    # Fill missing features
    for col in system.features:
        if col not in engineered.columns:
            engineered[col] = 0.0
    engineered[system.features] = engineered[system.features].ffill().bfill().fillna(0)

    # Scale and predict on ALL rows
    X = system.scaler.transform(engineered[system.features].values)
    probas = system.model.predict_proba(X)[:, 1]

    # Build per-row results
    rows = []
    dates = engineered["Date"].dt.strftime("%Y-%m-%d").tolist() if "Date" in engineered.columns else [str(i) for i in range(len(engineered))]
    closes = engineered["Close"].tolist() if "Close" in engineered.columns else [None] * len(engineered)

    for i in range(len(engineered)):
        prob = float(probas[i])
        if prob > buy_thresh:
            action = "BUY"
        elif prob < sell_thresh:
            action = "SELL"
        else:
            action = "WAIT"

        confidence_margin = abs(prob - 0.5) * 100
        rows.append({
            "date": dates[i],
            "close": round(closes[i], 2) if closes[i] is not None else None,
            "probability": round(prob * 100, 2),
            "action": action,
            "confidence_margin": round(confidence_margin, 2),
        })

    # Latest prediction (most recent row)
    latest = rows[-1] if rows else None

    # Summary stats
    buy_count = sum(1 for r in rows if r["action"] == "BUY")
    sell_count = sum(1 for r in rows if r["action"] == "SELL")
    wait_count = sum(1 for r in rows if r["action"] == "WAIT")
    avg_prob = float(np.mean(probas)) * 100

    summary = {
        "total_rows": len(rows),
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "wait_signals": wait_count,
        "avg_probability": round(avg_prob, 2),
        "latest_prediction": latest,
        "target_horizon": target_horizon,
        "target_threshold": target_threshold,
        "buy_threshold": buy_thresh,
        "sell_threshold": sell_thresh,
        "fee_pct": fee_pct,
    }

    return rows, summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main page."""
    # Default configuration values from meta.json
    config = {
        "target_horizon": 10,
        "target_threshold": 1.5,
        "buy_threshold": 51.0,
        "sell_threshold": 41.0,
        "fee_pct": 0.3,
    }
    if MODEL_DIR.exists() and (MODEL_DIR / "meta.json").exists():
        with open(MODEL_DIR / "meta.json") as f:
            meta = json.load(f)
            config["target_horizon"] = meta.get("target_horizon", 10)
            config["target_threshold"] = meta.get("target_threshold", 0.015) * 100
            config["buy_threshold"] = meta.get("thresholds", {}).get("buy", 0.51) * 100
            config["sell_threshold"] = meta.get("thresholds", {}).get("sell", 0.41) * 100
            config["fee_pct"] = meta.get("fee", 0.003) * 100

    model_loaded = system is not None and system.model is not None
    return render_template("index.html", config=config, model_loaded=model_loaded)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle CSV upload and return predictions."""
    try:
        # Get configuration from form
        buy_thresh = float(request.form.get("buy_threshold", 51.0)) / 100.0
        sell_thresh = float(request.form.get("sell_threshold", 41.0)) / 100.0
        target_horizon = int(request.form.get("target_horizon", 10))
        target_threshold = float(request.form.get("target_threshold", 1.5)) / 100.0
        fee_pct = float(request.form.get("fee_pct", 0.3)) / 100.0

        # Check file upload
        if "csv_file" not in request.files:
            return jsonify({"error": "No CSV file uploaded"}), 400

        file = request.files["csv_file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a CSV file"}), 400

        # Save and read CSV
        filepath = UPLOAD_FOLDER / file.filename
        file.save(str(filepath))

        df = pd.read_csv(str(filepath))

        # Validate required columns
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            return jsonify({"error": f"CSV missing required columns: {', '.join(missing)}"}), 400

        # Run predictions
        rows, summary = predict_from_df(df, buy_thresh, sell_thresh, target_horizon, target_threshold, fee_pct)

        if rows is None:
            return jsonify({"error": summary}), 500

        # Return last 50 rows for display (most recent) + full summary
        display_rows = rows[-50:]

        return jsonify({
            "success": True,
            "summary": summary,
            "predictions": display_rows,
            "filename": file.filename,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_all", methods=["POST"])
def predict_all():
    """Return all predictions (for download/full view)."""
    try:
        buy_thresh = float(request.form.get("buy_threshold", 51.0)) / 100.0
        sell_thresh = float(request.form.get("sell_threshold", 41.0)) / 100.0
        target_horizon = int(request.form.get("target_horizon", 10))
        target_threshold = float(request.form.get("target_threshold", 1.5)) / 100.0
        fee_pct = float(request.form.get("fee_pct", 0.3)) / 100.0

        if "csv_file" not in request.files:
            return jsonify({"error": "No CSV file uploaded"}), 400

        file = request.files["csv_file"]
        filepath = UPLOAD_FOLDER / file.filename
        file.save(str(filepath))
        df = pd.read_csv(str(filepath))

        rows, summary = predict_from_df(df, buy_thresh, sell_thresh, target_horizon, target_threshold, fee_pct)
        if rows is None:
            return jsonify({"error": summary}), 500

        return jsonify({"success": True, "summary": summary, "predictions": rows})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
