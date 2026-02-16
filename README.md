# Machine Learning-Driven Digital Asset Analytics System

This is a containerized system composed of a Spring Boot application, two FastAPI microservices, 
and a PostgreSQL database. It integrates historical market data and news sentiment analysis to provide 
machine-learning insights for digital assets.

---

## Technologies Used

- Java Spring Boot
- Python FastAPI
- PostgreSQL
- Docker & Docker Compose

---

## Architecture

The system consists of four services:

- A **Spring Boot** application acting as the main backend and entry point
- A **FastAPI LSTM microservice** for machine-learning predictions
- A **FastAPI NLP microservice** for news sentiment analysis
- A **PostgreSQL** database for persistent storage

Services communicate internally through Docker’s network.

---

## Exposed Ports

| Service  | Technology   | Port |
|----------|--------------|------|
| spring   | Spring Boot  | 8080 |
| lstm     | FastAPI      | 8000 |
| nlp      | FastAPI      | 8001 |
| postgres | PostgreSQL   | 5432 |

---

## Service Description

### Spring Boot (8080)
Acts as the main application of the system.  
Handles client requests and communicates with the FastAPI microservices.

### LSTM Microservice (8000)
A Long Short-Term Memory (LSTM) neural network is trained as a univariate time-series regression problem, where sequences of past closing prices are used to predict the next time step.

A sliding window (lookback) approach is applied, and the network consists of stacked LSTM layers followed by a fully connected output layer.

Exposes REST endpoints used by the Spring application.

### NLP Microservice (8001)
News sentiment analysis is performed using a pretrained transformer natural language processing model.

Recent cryptocurrency news articles are retrieved from the database and analyzed using a DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2).

Each article is classified as positive or negative, along with a confidence score.

Exposes REST endpoints used by the Spring application.

### PostgreSQL (5432)
Relational database used for storing application data.

---

## API Documentation

Each service exposes interactive documentation:

| Service         | Port | Documentation |
|----------------|------|---------------|
| LSTM | 8000 | `/docs` |
| NLP | 8001 | `/docs` |

---

## Running the Project

```bash
docker compose up --build
