.PHONY: install test lint lint-fix run-backend run-frontend clean

install:
	cd backend && pip install -r requirements.txt && pip install -r requirements-dev.txt
	cd frontend && npm install

test:
	cd backend && pytest
	cd frontend && npm test

lint:
	cd backend && ruff check . && ruff format --check .
	cd frontend && npm run lint

lint-fix:
	cd backend && ruff check . --fix && ruff format .
	cd frontend && npm run lint -- --fix

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
