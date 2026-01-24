FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi  \
        scikit-learn  \
        uvicorn \
        pandas \
        numpy \
        psycopg2-binary

COPY analytics/python /app/

EXPOSE 8001

CMD ["uvicorn", "tech.app:app", "--host", "0.0.0.0", "--port", "8001"]
