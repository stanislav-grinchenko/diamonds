FROM python:3.11-slim
WORKDIR /
COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
RUN pip install --no-cache-dir .
COPY models/ models/

CMD ["uvicorn", "src.diamonds.api:app", "--host", "0.0.0.0", "--port", "8000"]