FROM python:3.11-slim

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install \
        fastapi \
        uvicorn \
        pandas \
        numpy \
        psycopg2-binary \
        torch \
        transformers

COPY analytics/python /app/

EXPOSE 8001

CMD ["uvicorn", "nlp.app:app", "--host", "0.0.0.0", "--port", "8001"]