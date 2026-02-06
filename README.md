# Microservices with Docker

This project is a containerized full-stack system composed of a Spring MVC application, three FastAPI microservices, and a PostgreSQL database. All services are orchestrated using Docker Compose.

---

## Technologies Used

- Java Spring MVC
- Python FastAPI
- PostgreSQL
- Docker & Docker Compose

---

## Architecture

The system consists of five services:

- A **Spring MVC** application acting as the main backend and entry point
- A **FastAPI LSTM microservice** for machine-learning predictions
- A **FastAPI Technical Analysis microservice**
- A **FastAPI NLP microservice** for news sentiment analysis
- A **PostgreSQL** database for persistent storage

Services communicate internally through Docker’s network.

---

## Exposed Ports

| Service  | Technology   | Port |
|----------|--------------|------|
| spring   | Spring MVC   | 8080 |
| lstm     | FastAPI      | 8000 |
| tech     | FastAPI      | 8001 |
| nlp      | FastAPI      | 8002 |
| postgres | PostgreSQL   | 5432 |

---

## Service Description

### Spring MVC (8080)
The main backend application.  
Handles client requests and communicates with the FastAPI microservices.

### LSTM Microservice (8000)
FastAPI service responsible for LSTM prediction logic.  
The neural network is trained as a univariate time-series regression problem based on past closing prices.  
Exposes REST endpoints used by the Spring application.

### Technical Microservice (8001)
FastAPI service providing technical analysis functionality.  
Exposes REST endpoints used by the Spring application.

### NLP Microservice (8002)
FastAPI service responsible for transformer-based sentiment analysis of cryptocurrency news.  
Uses a pretrained DistilBERT model to classify articles as positive or negative.  
Exposes REST endpoints used by the Spring application.

### PostgreSQL (5432)
Relational database used for storing application data.

---

## Running the Project

```bash
docker compose up --build
