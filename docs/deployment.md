# Deployment Guide

This guide provides instructions for setting up and deploying the Unified ML Platform in various environments.

## Local Development Setup

### Prerequisites
- Python 3.9+
- Node.js 18+ and npm
- Make (optional, but recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harshjanghu24/Unified-Ml-platform.git
   cd Unified-Ml-platform
   ```

2. **Install dependencies:**
   Using Make:
   ```bash
   make install
   ```
   Manual:
   ```bash
   cd backend && pip install -r requirements.txt
   cd ../frontend && npm install
   ```

### Running the Application

1. **Start the Backend:**
   Using Make:
   ```bash
   make run-backend
   ```
   Manual:
   ```bash
   cd backend && uvicorn app.main:app --reload
   ```

2. **Start the Frontend:**
   Using Make:
   ```bash
   make run-frontend
   ```
   Manual:
   ```bash
   cd frontend && npm run dev
   ```

The application will be available at `http://localhost:5173`.

## Configuration

The backend can be configured using environment variables or a `.env` file in the `backend/` directory.

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Name of the application | `Unified ML Platform` |
| `DB_PATH` | Path to the SQLite database | `data/platform.db` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) | `INFO` |

## Production Deployment

### Backend (Uvicorn/Gunicorn)
For production, it is recommended to use `gunicorn` with the `uvicorn` worker class for better process management:
```bash
cd backend
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

### Frontend (Static Hosting)
Build the frontend for production:
```bash
cd frontend
npm run build
```
The resulting `dist/` folder can be served by any static file host or a web server like Nginx.

## Docker Deployment (Optional)
While a Dockerfile is not provided in the root, you can containerize the application by creating standard Dockerfiles.

**Example: Backend `Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directory for data storage
RUN mkdir -p data/uploads data/models data/reports

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
