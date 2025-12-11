import os
from pathlib import Path
from fastapi import UploadFile, HTTPException
import uuid

class FileHandler:
    """Simple file upload and cleanup"""
    
    ALLOWED_EXTENSIONS = {'.pdf'}
    UPLOAD_DIR = Path("uploads")
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self):
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        print(f"[OK] Upload directory: {self.UPLOAD_DIR.absolute()}")
    
    async def save_upload(self, file: UploadFile) -> str:
        """Save uploaded file"""
        # Validate extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Only PDF allowed."
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = self.UPLOAD_DIR / unique_filename
        
        # Save file
        try:
            with open(file_path, "wb") as buffer:
                total_size = 0
                while chunk := await file.read(8192):
                    total_size += len(chunk)
                    
                    if total_size > self.MAX_FILE_SIZE:
                        os.remove(file_path)
                        raise HTTPException(
                            status_code=400,
                            detail=f"File too large. Max: 50MB"
                        )
                    
                    buffer.write(chunk)
            
            return str(file_path)
            
        except Exception as e:
            if file_path.exists():
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def cleanup(self, file_path: str):
        """Remove temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[OK] Cleaned up: {file_path}")
        except Exception as e:
            print(f"[!] Cleanup warning: {e}")