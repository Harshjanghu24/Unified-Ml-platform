.PHONY: install test lint run-backend run-frontend clean

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

test:
	cd backend && pytest
	cd frontend && npm test

lint:
	cd backend && ruff check . && black --check .
	cd frontend && npm run lint

run-backend:
	cd backend && uvicorn app.main:app --reload

run-frontend:
	cd frontend && npm run dev

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf frontend/dist
	rm -rf frontend/node_modules
