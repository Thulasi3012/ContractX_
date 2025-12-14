import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Rate Limiting Configuration
    GEMINI_REQUEST_DELAY = float(os.getenv("GEMINI_REQUEST_DELAY", "2.0"))  # seconds between requests
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "60"))  # initial retry delay in seconds
    EXPONENTIAL_BACKOFF = os.getenv("EXPONENTIAL_BACKOFF", "true").lower() == "true"
    
    #Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")  # default = Docker service name
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

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
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "true").lower() == "true"
    
    @property
    def DATABASE_URL(self) -> str:
        """Create properly escaped database URL"""
        # Escape special characters in password
        import urllib.parse
        escaped_password = urllib.parse.quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{escaped_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
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