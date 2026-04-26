FROM python:3.11-slim

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py train.py entrypoint.sh ./
COPY static/ static/

RUN chmod +x entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
