# Contributing to Unified ML Platform
We love your input! We want to make contributing to this project as easy and transparent as possible. Lead maintainer: @Harshjanghu24.


## Technical Setup

### Prerequisites
- Python 3.9 or higher
- Node.js 18 or higher
- npm 9 or higher

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest ruff black
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run development server:
   ```bash
   npm run dev
   ```
4. Build for production:
   ```bash
   npm run build
   ```

## Development Workflow

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints and is formatted correctly.
6. Issue that pull request!

### Running Tests

#### Backend Tests
We use `pytest` for backend testing.
```bash
cd backend
pytest
```

#### Frontend Tests
We use `vitest` and `React Testing Library` for frontend testing.
```bash
cd frontend
npm test
```
*Note: The frontend test suite is currently being established.*

### Linting and Formatting
We use `ruff` for linting and `black` for formatting.

**Check for linting issues:**
```bash
ruff check .
```

**Format code:**
```bash
black .
```

## Pull Request Process
1. Update the `README.md` or other documentation with details of changes to the interface, this includes new environment variables, exposed ports, or location of database files.
2. Update the `CHANGELOG.md` with a summary of your changes under the `[Unreleased]` section.
3. The PR will be merged once it has been reviewed and approved by at least one maintainer.
