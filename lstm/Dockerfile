FROM python:3.11-slim

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install \
        fastapi \
        uvicorn \
        pandas \
        numpy \
        scikit-learn \
        psycopg2-binary \
        torch

COPY analytics/python /app/

EXPOSE 8000
CMD ["uvicorn", "lstm.app:app", "--host", "0.0.0.0", "--port", "8000"]
