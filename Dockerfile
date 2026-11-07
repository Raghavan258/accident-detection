FROM python:3.11-slim

WORKDIR /app
COPY requirements-serve.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-serve.txt
ENV PYTHONPATH=src
ENV MLFLOW_TRACKING_URI=sqlite:///mlops/mlflow.db
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "accident_detection.mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
