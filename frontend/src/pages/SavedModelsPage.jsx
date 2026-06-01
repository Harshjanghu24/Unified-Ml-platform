import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiSave, HiTrash } from 'react-icons/hi';
import { listModels, deleteModel, generateReport } from '../services/api.js';

export default function SavedModelsPage() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const { data } = await listModels();
      setModels(data.models || []);
    } catch {
      toast.error('Failed to load models.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleDelete = async (id, name) => {
    if (!confirm(`Delete model "${name}"?`)) return;
    try {
      await deleteModel(id);
      toast.success(`Model "${name}" deleted.`);
      fetchModels();
    } catch {
      toast.error('Delete failed.');
    }
  };

  const downloadReport = async () => {
    try {
      const res = await generateReport();
      const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'ml_report.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      toast.success('Report downloaded!');
    } catch {
      toast.error('Report generation failed. Train models first.');
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '80px' }}>
        <div className="spinner" style={{ margin: '0 auto 16px' }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading saved models...</p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px', marginBottom: '24px' }}>
          <div className="section-header" style={{ marginBottom: 0 }}>
            <div className="icon gradient-accent" style={{ color: 'white' }}>
              <HiSave />
            </div>
            <div>
              <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Saved Models</h1>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Manage your trained and persisted models
              </p>
            </div>
          </div>
          <button className="btn-primary" onClick={downloadReport}>
            📄 Download Full Report
          </button>
        </div>
      </motion.div>

      {models.length === 0 ? (
        <div className="glass-card" style={{ padding: '60px', textAlign: 'center' }}>
          <p style={{ fontSize: '3rem', marginBottom: '16px' }}>💾</p>
          <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>No Saved Models</h3>
          <p style={{ color: 'var(--text-muted)' }}>Train models to see them here.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '16px' }}>
          {models.map((model, i) => (
            <motion.div
              key={model.id}
              className="glass-card"
              style={{
                padding: '24px',
                border: model.is_best ? '1px solid var(--accent-amber)' : undefined,
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                    <h3 style={{ fontWeight: 700, fontSize: '1.1rem' }}>{model.model_name}</h3>
                    <span className={`badge badge-${model.problem_type}`}>
                      {model.problem_type}
                    </span>
                    {model.is_best ? <span className="badge badge-best">⭐ Best</span> : null}
                  </div>

                  {/* Metrics */}
                  <div className="metrics-grid" style={{ marginBottom: '12px' }}>
                    {Object.entries(model.metrics || {}).filter(([k, v]) =>
                      typeof v === 'number' && !['y_pred', 'y_test'].includes(k)
                    ).map(([key, value]) => (
                      <div key={key} style={{ padding: '8px 12px', background: 'var(--bg-primary)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div style={{ fontWeight: 700, fontSize: '1rem' }}>{value.toFixed(4)}</div>
                      </div>
                    ))}
                  </div>

                  {/* Meta */}
                  <div style={{ display: 'flex', gap: '24px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    <span>⏱ {model.training_time}s</span>
                    <span>📅 {new Date(model.created_at).toLocaleDateString()}</span>
                    {model.best_params && Object.keys(model.best_params).length > 0 && (
                      <span>⚙️ {Object.entries(model.best_params).map(([k, v]) => `${k}=${v}`).join(', ')}</span>
                    )}
                  </div>

                  {/* CV Scores */}
                  {model.cv_scores && Object.keys(model.cv_scores).length > 0 && (
                    <div style={{ marginTop: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      <strong>Cross Validation: </strong>
                      {Object.entries(model.cv_scores).map(([k, v]) => (
                        <span key={k} style={{ marginRight: '12px' }}>
                          {k}: {v.mean} ± {v.std}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <button
                  className="btn-danger"
                  onClick={() => handleDelete(model.id, model.model_name)}
                >
                  <HiTrash /> Delete
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
