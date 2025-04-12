import os
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import joblib
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Audio Fake/Real Classifier API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define available models
MODELS = {
    "2sec": {
        "model": "For_2sec_MLP_model.joblib",
        "scaler": "For_2sec_scaler.joblib",
        "encoder": "For_2sec_label_encoder.joblib",
        "duration": 2.0
    },
    "norm": {
        "model": "For_Norm_MLP_model.joblib",
        "scaler": "For_Norm_scaler.joblib", 
        "encoder": "For_Norm_label_encoder.joblib",
        "duration": None  # Will use full audio
    }
}

# Dictionary to hold loaded models
loaded_models = {}

# Load models function
def load_model(model_type):
    if model_type in loaded_models:
        return loaded_models[model_type]
    
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        model_info = MODELS[model_type]
        model = joblib.load(model_info["model"])
        scaler = joblib.load(model_info["scaler"])
        
        try:
            encoder = joblib.load(model_info["encoder"])
        except:
            encoder = LabelEncoder()
            encoder.fit(['fake', 'real'])  # Recreate with same order as training
        
        loaded_models[model_type] = {
            "model": model,
            "scaler": scaler,
            "encoder": encoder,
            "duration": model_info["duration"]
        }
        print(f"Model {model_type} loaded successfully")
        return loaded_models[model_type]
    except Exception as e:
        print(f"Error loading model {model_type}: {e}")
        raise ValueError(f"Failed to load model {model_type}: {str(e)}")

# Feature extraction functions
def extract_mfcc_features(audio_data, sample_rate, n_mfcc=40):
    mfccs = []
    audio_data = audio_data.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc).mean(axis=1)
    mfccs.append(mfcc)
    return np.array(mfccs)[0]

def extract_spectral_features(audio_data, sample_rate):
    audio_data = audio_data.astype(np.float32)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate).mean(axis=1)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).mean()
    return np.hstack([spectral_centroid, spectral_rolloff, spectral_contrast, spectral_bandwidth])

def extract_raw_signal_features(audio_data, sample_rate):
    audio_data = audio_data.astype(np.float32)
    zero_cross_rate = librosa.feature.zero_crossing_rate(y=audio_data).mean()
    signal_energy = np.sum(audio_data**2) / len(audio_data)
    return np.array([zero_cross_rate, signal_energy])

def combine_features(mfcc, spectral, raw):
    return np.hstack((mfcc, spectral, raw))

# Preprocess audio function
def preprocess_audio(audio_data, scaler, sample_rate=22050, target_duration=None):
    if target_duration:
        target_length = int(sample_rate * target_duration)
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        elif len(audio_data) > target_length:
            audio_data = audio_data[:target_length]

    audio_data = audio_data.astype(np.float32)
    mfcc_features = extract_mfcc_features(audio_data, sample_rate)
    spectral_features = extract_spectral_features(audio_data, sample_rate)
    raw_features = extract_raw_signal_features(audio_data, sample_rate)
    features = combine_features(mfcc_features, spectral_features, raw_features)
    features = features.reshape(1, -1)
    return scaler.transform(features)

# Prediction function
def predict_audio(audio_data, model_type="2sec"):
    model_data = load_model(model_type)
    model = model_data["model"]
    scaler = model_data["scaler"]
    duration = model_data["duration"]
    
    features = preprocess_audio(audio_data, scaler, target_duration=duration)
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    class_labels = {0: 'Fake', 1: 'Real'}
    predicted_class = class_labels[prediction]
    return predicted_class, probs, features[0]  # Return features for visualization

# Response model for each audio result
class AudioResult(BaseModel):
    filename: str
    model_used: str
    predicted_class: str
    fake_probability: float
    real_probability: float
    features: dict  # Audio features

# API endpoint to handle multiple audio uploads
@app.post("/predict", response_model=List[AudioResult])
async def predict_audios(
    files: List[UploadFile] = File(...),
    model_type: str = Form("2sec")  # Default to 2sec model
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")
    
    # Validate model type
    if model_type not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Choose from: {', '.join(MODELS.keys())}")
    
    try:
        # Ensure model is loaded
        load_model(model_type)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    results = []
    
    for file in files:
        try:
            # Create a temporary file to store audio content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
                # Write the audio content to the temporary file
                content = await file.read()
                temp.write(content)
                temp_path = temp.name
            
            # Load the audio file using librosa
            try:
                audio_data, sr = librosa.load(temp_path, sr=22050, mono=True, dtype=np.float32)
                os.unlink(temp_path)  # Clean up the temporary file
            except Exception as e:
                os.unlink(temp_path)  # Clean up the temporary file
                raise HTTPException(status_code=400, detail=f"Error loading audio file: {str(e)}")
            
            # Predict and get results
            predicted_class, probs, features = predict_audio(audio_data, model_type)
            
            # Extract some features for visualization
            feature_dict = {
                "MFCC Mean": float(np.mean(features[:40])),  # First 40 are MFCCs
                "Spectral Centroid": float(features[40]),
                "Spectral Rolloff": float(features[41]),
                "Zero Crossing Rate": float(features[-2]),
                "Signal Energy": float(features[-1])
            }
            
            # Prepare result
            result = AudioResult(
                filename=file.filename,
                model_used=model_type,
                predicted_class=predicted_class,
                fake_probability=float(probs[0] * 100),  # Convert to percentage
                real_probability=float(probs[1] * 100),  # Convert to percentage
                features=feature_dict
            )
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing {file.filename}: {str(e)}")
    
    return results

# Get available models endpoint
@app.get("/models")
async def get_models():
    return {"available_models": list(MODELS.keys())}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)