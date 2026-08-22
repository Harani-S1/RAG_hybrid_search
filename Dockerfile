FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --no-compile -r requirements.txt

COPY app ./app
COPY src ./src
COPY data/raw ./data/raw
COPY streamlit_app.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]