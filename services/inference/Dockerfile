FROM python:3.11-slim

WORKDIR /app

COPY services/inference/requirements.txt \
     /app/requirements.txt

RUN pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    -r requirements.txt

COPY services/inference/app \
     /app/app

COPY shared \
     /app/shared

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]