FROM python:3.11-slim

WORKDIR /app

COPY services/api/requirements.txt \
     /app/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY services/api/app \
     /app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]