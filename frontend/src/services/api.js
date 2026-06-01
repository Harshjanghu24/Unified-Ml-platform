import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 min timeout for training
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Dataset APIs ──
export const uploadDataset = (formData) =>
  api.post('/upload-dataset', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
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

// ── Training APIs ──
export const trainModels = () => api.post('/train');
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
