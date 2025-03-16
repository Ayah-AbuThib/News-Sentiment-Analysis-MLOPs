# File: src/config/settings.py
"""Central configuration management for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

class PathConfiguration:
    """Manages all system paths and directories."""
    
    RAW_DATA = PROJECT_ROOT / "data/raw/news.csv"
    PROCESSED_DATA = PROJECT_ROOT / "data/processed/cleaned_news.csv"
    MODEL_PATH = PROJECT_ROOT / "models/saved_models/finbert"
    MODEL_LOGS = PROJECT_ROOT / "models/model_logs"

class TrainingParameters:
    """Contains model training hyperparameters."""
    
    MODEL_NAME = "ProsusAI/finbert"
    MAX_SEQUENCE_LENGTH = 500
    BATCH_SIZE = 32
    EPOCHS = 1
    VALIDATION_SPLIT = 0.5
