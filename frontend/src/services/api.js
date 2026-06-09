import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 1_800_000, // 30 min timeout for large uploads & training
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Dataset APIs ──
export const uploadDataset = (formData, onUploadProgress) =>
  api.post('/upload-dataset', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    // Support upload progress tracking for large files
    onUploadProgress,
    // No timeout override — the global 30 min is sufficient for 10 GB
  });

export const analyzeColumns = () => api.post('/analyze-columns');

export const getTargetRecommendations = () => api.get('/target-recommendations');

export const selectTarget = (targetColumn) => {
  const formData = new FormData();
  formData.append('target_column', targetColumn);
  return api.post('/select-target', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const getDatasetSummary = () => api.get('/dataset-summary');

// ── Preprocessing APIs ──
export const preprocessAuto = () => api.post('/preprocess/auto');
export const preprocessManual = (config) => api.post('/preprocess/manual', config);
export const getPreprocessingReport = () => api.get('/preprocess/report');
export const getEncodingRecommendations = () => api.get('/preprocess/recommendations');

// ── Training APIs ──
export const trainModels = (params = {}) => api.post('/train', params);
export const getTrainStatus = (jobId) => api.get(`/train-status/${jobId}`);
export const getTrainResult = (jobId) => api.get(`/train-result/${jobId}`);
export const getTrainingOptions = () => api.get('/training-options');
export const getSystemInfo = () => api.get('/system-info');

export const getMetrics = () => api.get('/metrics');
export const getEDA = () => api.get('/eda');

// ── Prediction APIs ──
export const predict = (features, explain = false) =>
  api.post('/predict', { features, explain });

export const predictBatch = (formData) =>
  api.post('/predict-batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

// ── Model Management APIs ──
export const listModels = () => api.get('/models');
export const deleteModel = (id) => api.delete(`/model/${id}`);
export const generateReport = () =>
  api.post('/generate-report', null, { responseType: 'blob' });

export default api;
