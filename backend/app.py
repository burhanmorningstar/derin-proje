#python app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

from config import API_HOST, API_PORT, DEBUG, LOG_LEVEL, LOG_FORMAT
from services.model_service import ModelService

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Add file handler
file_handler = RotatingFileHandler(
    log_dir / "app.log",
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize model service
model_service = ModelService()

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        file = request.files.get('frames')
        if not file:
            logger.warning("No frame file provided in request")
            return jsonify({"error": "Frame eksik!"}), 400

        # Process image
        image = Image.open(file.stream).convert("RGB")
        
        # Get prediction
        result = model_service.predict(image)
        
        logger.info(f"Prediction successful: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Check API status."""
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "device": model_service.device.type
    })

if __name__ == '__main__':
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
