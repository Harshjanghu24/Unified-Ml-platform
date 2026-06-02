import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiCursorClick, HiUpload } from 'react-icons/hi';
import { predict, predictBatch, getMetrics } from '../services/api.js';

export default function PredictionPage() {
  const [features, setFeatures] = useState({});
  const [featureNames, setFeatureNames] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [explain, setExplain] = useState(false);
  const [batchFile, setBatchFile] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [mode, setMode] = useState('single'); // 'single' or 'batch'

  useEffect(() => {
    // Get feature names from metrics
    getMetrics().then(() => {
    }).catch(() => {});
  }, []);

  const addFeature = () => {
    const name = prompt('Enter feature name:');
    if (name && name.trim()) {
      setFeatures({ ...features, [name.trim()]: '' });
      setFeatureNames([...featureNames, name.trim()]);
    }
  };

  const handlePredict = async () => {
    const featureValues = {};
    for (const [k, v] of Object.entries(features)) {
      const num = parseFloat(v);
      featureValues[k] = isNaN(num) ? v : num;
    }

    setLoading(true);
    setResult(null);
    try {
      const { data } = await predict(featureValues, explain);
      setResult(data);
      toast.success('Prediction complete!');
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPredict = async () => {
    if (!batchFile) {
      toast.error('Please select a CSV file.');
      return;
    }

    setBatchLoading(true);
    setBatchResults(null);
    const formData = new FormData();
    formData.append('file', batchFile);

    try {
      const { data } = await predictBatch(formData);
      setBatchResults(data);
      toast.success(`Batch prediction complete! ${data.total_rows} rows processed.`);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Batch prediction failed.');
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="section-header">
          <div className="icon gradient-accent" style={{ color: 'white' }}>
            <HiCursorClick />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Prediction</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Make predictions using the best trained model
            </p>
          </div>
        </div>
      </motion.div>

      {/* Mode Toggle */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '24px' }}>
        <button
          onClick={() => setMode('single')}
          style={{
            padding: '10px 24px', borderRadius: '8px', cursor: 'pointer',
            border: mode === 'single' ? '1px solid var(--accent-indigo)' : '1px solid var(--border-color)',
            background: mode === 'single' ? 'rgba(99, 102, 241, 0.1)' : 'var(--bg-card)',
            color: mode === 'single' ? 'var(--accent-indigo)' : 'var(--text-secondary)',
            fontWeight: mode === 'single' ? 600 : 400, fontSize: '0.9rem',
          }}
        >
          📝 Single Prediction
        </button>
        <button
          onClick={() => setMode('batch')}
          style={{
            padding: '10px 24px', borderRadius: '8px', cursor: 'pointer',
            border: mode === 'batch' ? '1px solid var(--accent-indigo)' : '1px solid var(--border-color)',
            background: mode === 'batch' ? 'rgba(99, 102, 241, 0.1)' : 'var(--bg-card)',
            color: mode === 'batch' ? 'var(--accent-indigo)' : 'var(--text-secondary)',
            fontWeight: mode === 'batch' ? 600 : 400, fontSize: '0.9rem',
          }}
        >
          📄 Batch Prediction
        </button>
      </div>

      {/* Single Prediction */}
      {mode === 'single' && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontWeight: 700 }}>Enter Feature Values</h3>
              <button className="btn-secondary" onClick={addFeature} style={{ fontSize: '0.85rem' }}>
                + Add Feature
              </button>
            </div>

            {Object.keys(features).length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                <p style={{ fontSize: '2rem', marginBottom: '8px' }}>📝</p>
                <p>Click &quot;Add Feature&quot; to add feature fields for prediction.</p>
                <p style={{ fontSize: '0.85rem', marginTop: '4px' }}>
                  Enter the same feature names used in your training dataset.
                </p>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '12px' }}>
                {Object.entries(features).map(([key]) => (
                  <div key={key}>
                    <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                      {key}
                      <button
                        onClick={() => {
                          const updated = { ...features };
                          delete updated[key];
                          setFeatures(updated);
                          setFeatureNames(featureNames.filter(n => n !== key));
                        }}
                        style={{
                          background: 'none', border: 'none', color: 'var(--accent-rose)',
                          cursor: 'pointer', fontSize: '0.8rem',
                        }}
                      >
                        ✕
                      </button>
                    </label>
                    <input
                      className="form-input"
                      placeholder="Value"
                      value={features[key]}
                      onChange={(e) => setFeatures({ ...features, [key]: e.target.value })}
                    />
                  </div>
                ))}
              </div>
            )}

            {Object.keys(features).length > 0 && (
              <div style={{ marginTop: '20px', display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
                <button className="btn-primary" onClick={handlePredict} disabled={loading}>
                  {loading ? '⏳ Predicting...' : '🚀 Predict'}
                </button>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.9rem' }}>
                  <input
                    type="checkbox"
                    checked={explain}
                    onChange={(e) => setExplain(e.target.checked)}
                    style={{ accentColor: 'var(--accent-indigo)' }}
                  />
                  Include SHAP Explanation
                </label>
              </div>
            )}
          </div>

          {/* Prediction Result */}
          {result && (
            <motion.div
              className="glass-card"
              style={{ padding: '24px' }}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>
                🎯 Prediction Result
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px', marginBottom: '16px' }}>
                <div className="stat-card" style={{ background: 'var(--bg-primary)', borderRadius: '10px', padding: '20px' }}>
                  <div className="stat-value gradient-text" style={{ fontSize: '1.5rem' }}>
                    {result.predicted_class ?? result.predicted_value}
                  </div>
                  <div className="stat-label">
                    {result.predicted_class !== undefined ? 'Predicted Class' : 'Predicted Value'}
                  </div>
                </div>

                {result.confidence !== undefined && (
                  <div className="stat-card" style={{ background: 'var(--bg-primary)', borderRadius: '10px', padding: '20px' }}>
                    <div className="stat-value" style={{ color: 'var(--accent-emerald)', fontSize: '1.5rem' }}>
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="stat-label">Confidence</div>
                  </div>
                )}

                <div className="stat-card" style={{ background: 'var(--bg-primary)', borderRadius: '10px', padding: '20px' }}>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>{result.model_used}</div>
                  <div className="stat-label">Model Used</div>
                </div>
              </div>

              {/* Class Probabilities */}
              {result.class_probabilities && (
                <div style={{ marginBottom: '16px' }}>
                  <h4 style={{ fontWeight: 600, marginBottom: '8px', fontSize: '0.95rem' }}>Class Probabilities</h4>
                  <div style={{ display: 'grid', gap: '6px' }}>
                    {Object.entries(result.class_probabilities).map(([cls, prob]) => (
                      <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span style={{ minWidth: '80px', fontSize: '0.85rem' }}>{cls}</span>
                        <div style={{
                          flex: 1, height: '8px', background: 'var(--bg-primary)',
                          borderRadius: '4px', overflow: 'hidden',
                        }}>
                          <div style={{
                            width: `${prob * 100}%`, height: '100%',
                            background: 'var(--gradient-accent)', borderRadius: '4px',
                          }} />
                        </div>
                        <span style={{ fontWeight: 600, fontSize: '0.85rem', minWidth: '45px' }}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* SHAP Explanation */}
              {result.explanation?.contributions && result.explanation.contributions.length > 0 && (
                <div>
                  <h4 style={{ fontWeight: 600, marginBottom: '8px', fontSize: '0.95rem' }}>
                    🔍 Why This Prediction? (SHAP)
                  </h4>
                  <div style={{ display: 'grid', gap: '4px' }}>
                    {result.explanation.contributions.map((c, i) => (
                      <div key={i} style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        padding: '6px 12px', background: 'var(--bg-primary)',
                        borderRadius: '6px', fontSize: '0.85rem',
                      }}>
                        <span style={{ flex: 1, color: 'var(--text-secondary)' }}>{c.feature}</span>
                        <span style={{
                          color: c.shap_value > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                          fontWeight: 600,
                        }}>
                          {c.shap_value > 0 ? '+' : ''}{c.shap_value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </motion.div>
      )}

      {/* Batch Prediction */}
      {mode === 'batch' && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
            <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>Upload CSV for Batch Prediction</h3>
            <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
              <label
                style={{
                  padding: '10px 24px', borderRadius: '8px',
                  border: '1px dashed var(--border-color)', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: '8px',
                  color: 'var(--text-secondary)', fontSize: '0.9rem',
                }}
              >
                <HiUpload /> {batchFile ? batchFile.name : 'Choose CSV File'}
                <input
                  type="file"
                  accept=".csv"
                  style={{ display: 'none' }}
                  onChange={(e) => setBatchFile(e.target.files[0])}
                />
              </label>
              <button className="btn-primary" onClick={handleBatchPredict} disabled={batchLoading || !batchFile}>
                {batchLoading ? '⏳ Processing...' : '🚀 Predict All'}
              </button>
            </div>
          </div>

          {/* Batch Results */}
          {batchResults && (
            <div className="glass-card" style={{ padding: '24px' }}>
              <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>
                Batch Results ({batchResults.total_rows} rows)
              </h3>
              <div style={{ overflowX: 'auto', maxHeight: '500px', overflowY: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Row</th>
                      <th>{batchResults.problem_type === 'regression' ? 'Predicted Value' : 'Predicted Class'}</th>
                      {batchResults.predictions[0]?.confidence !== undefined && <th>Confidence</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.predictions.map((p, i) => (
                      <tr key={i}>
                        <td>{p.row}</td>
                        <td style={{ fontWeight: 600 }}>
                          {p.predicted_class ?? p.predicted_value ?? p.error}
                        </td>
                        {p.confidence !== undefined && (
                          <td>{(p.confidence * 100).toFixed(1)}%</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
