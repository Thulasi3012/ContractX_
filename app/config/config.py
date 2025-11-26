import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Rate Limiting Configuration
    GEMINI_REQUEST_DELAY = float(os.getenv("GEMINI_REQUEST_DELAY", "2.0"))  # seconds between requests
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "60"))  # initial retry delay in seconds
    EXPONENTIAL_BACKOFF = os.getenv("EXPONENTIAL_BACKOFF", "true").lower() == "true"
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "52428800"))  # 50MB
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
    ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx'}
    
    # Logging Configuration
    LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Configuration
    TABLE_DETECTION_MODEL = os.getenv("TABLE_DETECTION_MODEL", "microsoft/table-transformer-detection")
    TABLE_CONFIDENCE_THRESHOLD = float(os.getenv("TABLE_CONFIDENCE_THRESHOLD", "0.7"))
    
    # Processing Configuration
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "5"))
    DEFAULT_OVERLAP = int(os.getenv("DEFAULT_OVERLAP", "1"))
    
    # Create necessary directories
    @classmethod
    def initialize(cls):
        """Create required directories"""
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)
        print(f"✓ Upload directory: {cls.UPLOAD_DIR.absolute()}")
        print(f"✓ Log directory: {cls.LOG_DIR.absolute()}")
        
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables!")
        print(f"✓ Gemini API Key configured")
        print(f"✓ Model: {cls.GEMINI_MODEL}")
        print(f"✓ Request Delay: {cls.GEMINI_REQUEST_DELAY}s")
        print(f"✓ Max Retries: {cls.MAX_RETRIES}")

# Initialize on import
Config.initialize()