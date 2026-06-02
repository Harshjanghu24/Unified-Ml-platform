# Troubleshooting Guide

This guide covers common issues encountered while setting up or using the Unified ML Platform and provides steps to resolve them.

## Installation Issues

### Missing Python Dependencies
**Problem:** `ImportError` or `ModuleNotFoundError` when starting the backend.
**Solution:** Ensure all dependencies are installed in your active environment:
```bash
pip install -r backend/requirements.txt
```

### Node.js Versions
**Problem:** `npm install` fails or `npm run dev` throws errors related to dependencies.
**Solution:** Ensure you are using a compatible Node.js version (18+ recommended). Use `nvm` to manage Node.js versions if necessary.

## Runtime Issues

### CORS Errors
**Problem:** The frontend cannot communicate with the backend, and the browser console shows CORS errors.
**Solution:** Check `backend/app/main.py` for allowed origins. Ensure `http://localhost:5173` is included. If running on a different port, update the `allow_origins` list.

### Database Initialization Failure
**Problem:** Backend fails to start with errors related to `platform.db`.
**Solution:** Ensure the `backend/data/` directory exists and has write permissions. The application attempts to create the database automatically on startup.

### Training Timeouts
**Problem:** The frontend stops polling or shows a timeout error during long training sessions.
**Solution:** For very large datasets, training can take several minutes. The platform uses asynchronous background tasks, so you can safely refresh the page; the state will be restored from the backend.

## Performance Issues

### Memory Spikes
**Problem:** The backend crashes or slows down significantly during training.
**Solution:** The platform has built-in guards for high-cardinality columns and large datasets. However, if you encounter issues, try reducing the dataset size or the number of features before uploading.

### GPU Not Detected
**Problem:** Training is slow and does not seem to use the GPU despite having one.
**Solution:** Ensure NVIDIA drivers and CUDA are correctly installed. The backend uses `nvidia-smi` to detect GPU status. XGBoost will automatically fall back to CPU if a compatible GPU is not found.

## Error Logs
For more detailed information, check the console output where the backend is running. You can increase the log level by setting `LOG_LEVEL=DEBUG` in your environment.
