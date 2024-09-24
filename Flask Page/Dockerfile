FROM python:3.9-slim
   # Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Use default port 8000 if PORT is not set
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} app:app