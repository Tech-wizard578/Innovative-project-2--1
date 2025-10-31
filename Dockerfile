# SmartSched - AI-Powered Process Scheduler
# Docker Container for Easy Deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data logs results

# Train models on container startup (optional - can be skipped if models exist)
# Uncomment this line to auto-train on first run:
# RUN python train_models.py --quick

# Expose Flask port
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/presets')"

# Default command - run Flask app
CMD ["python", "app.py"]

# Alternative commands (can be overridden):
# docker run smartsched python main_demo.py
# docker run smartsched python train_models.py
# docker run smartsched python demo_borg.py