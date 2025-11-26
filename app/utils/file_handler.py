import os
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from typing import Optional
import uuid

class FileHandler:
    """Handles file upload, validation, and cleanup"""
    
    ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx'}
    UPLOAD_DIR = Path("uploads")
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self):
        self.UPLOAD_DIR.mkdir(exist_ok=True)
    
    async def save_upload(self, file: UploadFile) -> str:
        """
        Save uploaded file to disk
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            Path to saved file
            
        Raises:
            HTTPException: If file validation fails
        """
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {self.ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = self.UPLOAD_DIR / unique_filename
        
        # Save file in chunks to handle large files
        try:
            with open(file_path, "wb") as buffer:
                total_size = 0
                while chunk := await file.read(8192):  # 8KB chunks
                    total_size += len(chunk)
                    
                    # Check file size
                    if total_size > self.MAX_FILE_SIZE:
                        os.remove(file_path)
                        raise HTTPException(
                            status_code=400,
                            detail=f"File too large. Max size: {self.MAX_FILE_SIZE / (1024*1024)}MB"
                        )
                    
                    buffer.write(chunk)
            
            return str(file_path)
            
        except Exception as e:
            # Cleanup on error
            if file_path.exists():
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    async def cleanup(self, file_path: str) -> None:
        """
        Remove temporary files after processing
        
        Args:
            file_path: Path to file to remove
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Cleanup warning: {str(e)}")
    
    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """Check if file exists"""
        return os.path.exists(file_path)