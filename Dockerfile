FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first (cached)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app folder
COPY ./app ./app

# Expose port
EXPOSE 8000

# Run Uvicorn pointing to app/main.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
