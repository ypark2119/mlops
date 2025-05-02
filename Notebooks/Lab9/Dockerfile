FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY redditApp.py .
COPY reddit_model_pipeline.joblib .
COPY . .

EXPOSE 8000

CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]
