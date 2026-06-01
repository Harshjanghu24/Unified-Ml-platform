import { useState } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiBeaker, HiCheckCircle, HiCog } from 'react-icons/hi';
import { trainModels } from '../services/api.js';

export default function TrainingPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  const handleTrain = async () => {
    setLoading(true);
    setResult(null);
    setElapsed(0);

    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);

    try {
      const { data } = await trainModels();
      setResult(data);
      toast.success(`Training complete! Best model: ${data.best_model}`);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Training failed. Make sure a dataset is uploaded.');
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="section-header">
          <div className="icon gradient-accent" style={{ color: 'white' }}>
            <HiBeaker />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Model Training</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Train models with automatic hyperparameter tuning and cross-validation
            </p>
          </div>
        </div>
      </motion.div>

      {/* Train Button */}
      <motion.div
        className="glass-card"
        style={{ padding: '40px', textAlign: 'center', marginBottom: '24px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        {!loading && !result && (
          <>
            <p style={{ fontSize: '3rem', marginBottom: '16px' }}>🧠</p>
            <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '1.2rem' }}>
              Ready to Train
            </h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '24px', maxWidth: '500px', margin: '0 auto 24px' }}>
              This will preprocess your data, run feature selection, train multiple models
              with hyperparameter tuning (GridSearchCV), perform 5-Fold Cross Validation,
              and generate SHAP explanations.
            </p>
            <button className="btn-primary" onClick={handleTrain} style={{ fontSize: '1rem', padding: '14px 40px' }}>
              <HiBeaker /> Start Training Pipeline
            </button>
          </>
        )}

        {loading && (
          <>
            <div className="spinner pulse-glow" style={{ margin: '0 auto 20px', width: '60px', height: '60px', borderWidth: '4px' }} />
            <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Training in Progress...</h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '8px' }}>
              Preprocessing → Feature Selection → Model Training → Cross Validation → SHAP Analysis
            </p>
            <p style={{ color: 'var(--accent-indigo)', fontWeight: 600, fontSize: '1.2rem' }}>
              {elapsed}s elapsed
            </p>
          </>
        )}

        {!loading && result && (
          <>
            <HiCheckCircle style={{ fontSize: '3rem', color: 'var(--accent-emerald)', marginBottom: '12px' }} />
            <h3 style={{ fontWeight: 700, marginBottom: '8px', color: 'var(--accent-emerald)' }}>
              Training Complete!
            </h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '16px' }}>
              All models trained successfully. View results in the Evaluation page.
            </p>
            <button className="btn-primary" onClick={handleTrain}>
              <HiBeaker /> Re-Train
            </button>
          </>
        )}
      </motion.div>

      {/* Preprocessing Steps */}
      {result?.preprocessing_steps && (
        <motion.div
          className="glass-card"
          style={{ padding: '24px', marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3 style={{ fontWeight: 700, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <HiCog style={{ color: 'var(--accent-indigo)' }} /> Preprocessing Pipeline
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {result.preprocessing_steps.map((step, i) => (
              <div
                key={i}
                style={{
                  display: 'flex', alignItems: 'center', gap: '12px',
                  padding: '10px 16px', background: 'var(--bg-primary)',
                  borderRadius: '8px', fontSize: '0.9rem',
                }}
              >
                <span style={{
                  width: '24px', height: '24px', borderRadius: '50%',
                  background: 'var(--gradient-accent)', display: 'flex',
                  alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.7rem', fontWeight: 700, color: 'white', flexShrink: 0,
                }}>
                  {i + 1}
                </span>
                <span>{step}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Feature Selection Summary */}
      {result?.feature_selection && !result.feature_selection.error && (
        <motion.div
          className="glass-card"
          style={{ padding: '24px', marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>🎯 Feature Selection Results</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
            {result.feature_selection.random_forest_importance && !result.feature_selection.random_forest_importance.error && (
              <div>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '8px', color: 'var(--accent-indigo)' }}>
                  Random Forest Importance
                </h4>
                {result.feature_selection.random_forest_importance.slice(0, 8).map((f, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{f.feature}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{
                        width: `${Math.max(f.importance * 300, 4)}px`, height: '6px',
                        background: 'var(--gradient-accent)', borderRadius: '3px',
                      }} />
                      <span style={{ fontWeight: 600, minWidth: '45px', textAlign: 'right' }}>{f.importance}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {result.feature_selection.mutual_information && !result.feature_selection.mutual_information.error && (
              <div>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '8px', color: 'var(--accent-violet)' }}>
                  Mutual Information
                </h4>
                {result.feature_selection.mutual_information.slice(0, 8).map((f, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{f.feature}</span>
                    <span style={{ fontWeight: 600 }}>{f.mi_score}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Model Summary Cards */}
      {result?.models && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>📊 Trained Models</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
            {result.models.map((model, i) => (
              <div
                key={i}
                className="glass-card"
                style={{
                  padding: '20px',
                  border: model.is_best ? '2px solid var(--accent-amber)' : undefined,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <h4 style={{ fontWeight: 700, fontSize: '1rem' }}>{model.model_name}</h4>
                  {model.is_best && <span className="badge badge-best">⭐ Best</span>}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <p>Training Time: <strong>{model.training_time}s</strong></p>
                  {model.best_params && Object.keys(model.best_params).length > 0 && (
                    <p style={{ marginTop: '4px' }}>
                      Best Params: <code style={{ background: 'var(--bg-primary)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.75rem' }}>
                        {Object.entries(model.best_params).map(([k, v]) => `${k}=${v}`).join(', ')}
                      </code>
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
