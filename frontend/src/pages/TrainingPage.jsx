import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  HiBeaker, HiCheckCircle, HiCog, HiServer,
  HiAdjustments, HiSearch, HiViewGrid
} from 'react-icons/hi';
import {
  trainModels, getTrainStatus, getTrainResult,
  getSystemInfo, getTrainingOptions
} from '../services/api.js';

export default function TrainingPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [sysInfo, setSysInfo] = useState(null);

  // ── Manual Selection State ──
  const [trainingOptions, setTrainingOptions] = useState(null);
  const [optionsLoading, setOptionsLoading] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [featureSearch, setFeatureSearch] = useState('');

  // Fetch system info on mount
  useEffect(() => {
    const fetchSysInfo = async () => {
      try {
        const { data } = await getSystemInfo();
        setSysInfo(data);
      } catch (e) {
        console.error("Failed to load system info", e);
      }
    };
    fetchSysInfo();
    const sysTimer = setInterval(fetchSysInfo, 10000);
    return () => clearInterval(sysTimer);
  }, []);

  // Fetch training options (algorithms + features) on mount
  useEffect(() => {
    const fetchOptions = async () => {
      setOptionsLoading(true);
      try {
        const { data } = await getTrainingOptions();
        setTrainingOptions(data);
        // Default: select all features
        setSelectedFeatures(data.features.map(f => f.name));
      } catch (e) {
        console.error("Failed to load training options", e);
        // Not critical — user can still train with defaults
      } finally {
        setOptionsLoading(false);
      }
    };
    fetchOptions();
  }, []);

  // Filter features by search query
  const filteredFeatures = useMemo(() => {
    if (!trainingOptions?.features) return [];
    if (!featureSearch.trim()) return trainingOptions.features;
    const q = featureSearch.toLowerCase();
    return trainingOptions.features.filter(f =>
      f.name.toLowerCase().includes(q) || f.dtype.toLowerCase().includes(q)
    );
  }, [trainingOptions, featureSearch]);

  const toggleFeature = (featureName) => {
    setSelectedFeatures(prev =>
      prev.includes(featureName)
        ? prev.filter(f => f !== featureName)
        : [...prev, featureName]
    );
  };

  const selectAllFeatures = () => {
    if (trainingOptions?.features) {
      setSelectedFeatures(trainingOptions.features.map(f => f.name));
    }
  };

  const deselectAllFeatures = () => {
    setSelectedFeatures([]);
  };

  const handleTrain = async () => {
    // Validate at least one feature is selected when features are available
    if (trainingOptions?.features && selectedFeatures.length === 0) {
      toast.error('Please select at least one feature to train on.');
      return;
    }

    setLoading(true);
    setResult(null);
    setElapsed(0);
    setProgressText('Initializing pipeline...');

    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);

    // Build request payload
    const payload = {};
    if (selectedAlgorithm) {
      payload.algorithm = selectedAlgorithm;
    }
    if (trainingOptions?.features && selectedFeatures.length < trainingOptions.features.length) {
      // Only send features if user deselected some (optimization)
      payload.features = selectedFeatures;
    }

    try {
      const { data } = await trainModels(payload);
      const jobId = data.job_id;

      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await getTrainStatus(jobId);
          if (statusRes.data.status === 'completed') {
            clearInterval(pollInterval);
            clearInterval(timer);
            const resultRes = await getTrainResult(jobId);
            setResult(resultRes.data);
            toast.success(`Training complete! Best model: ${resultRes.data.best_model}`);
            setLoading(false);
          } else if (statusRes.data.status === 'failed') {
            clearInterval(pollInterval);
            clearInterval(timer);
            toast.error(statusRes.data.error || 'Training failed.');
            setLoading(false);
          } else {
            setProgressText(statusRes.data.progress || 'Running...');
          }
        } catch {
            clearInterval(pollInterval);
            clearInterval(timer);
            toast.error('Failed to get training status.');
            setLoading(false);
        }
      }, 2000);

    } catch (err) {
      clearInterval(timer);
      setLoading(false);
      toast.error(err.response?.data?.detail || 'Training failed to start.');
    }
  };

  const allFeaturesSelected = trainingOptions?.features &&
    selectedFeatures.length === trainingOptions.features.length;

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
              Select an algorithm and features, then train with automatic hyperparameter tuning
            </p>
          </div>
        </div>
      </motion.div>

      {/* System Info */}
      {sysInfo && (
        <motion.div
          className="glass-card"
          style={{ padding: '20px', marginBottom: '24px', display: 'flex', gap: '20px', alignItems: 'center' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <HiServer style={{ fontSize: '2rem', color: 'var(--accent-indigo)' }} />
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', flex: 1 }}>
            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>CPU</span>
              <div style={{ fontWeight: 600 }}>{sysInfo.cpu}</div>
            </div>
            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>RAM</span>
              <div style={{ fontWeight: 600 }}>{sysInfo.ram}</div>
            </div>
            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>GPU</span>
              <div style={{ fontWeight: 600 }}>{sysInfo.gpu}</div>
            </div>
            {sysInfo.cuda_status && (
              <div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>VRAM</span>
                <div style={{ fontWeight: 600 }}>{sysInfo.vram}</div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* ═══════════════════════════════════════════════════════
          Training Configuration Panel
          ═══════════════════════════════════════════════════════ */}
      {!loading && !result && trainingOptions && (
        <motion.div
          className="training-config"
          style={{ marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          {/* ── Algorithm Selection ── */}
          <div className="config-section">
            <div className="config-section-header">
              <div className="config-icon gradient-accent" style={{ color: 'white' }}>
                <HiAdjustments />
              </div>
              <div>
                <h4>Algorithm Selection</h4>
                <p>
                  Choose a specific algorithm or leave blank to auto-train all
                  {trainingOptions.problem_type && (
                    <> &nbsp;·&nbsp; Problem type: <strong style={{ color: 'var(--accent-indigo)' }}>
                      {trainingOptions.problem_type.toUpperCase()}
                    </strong></>
                  )}
                </p>
              </div>
            </div>

            <div className="algo-selector" id="algorithm-selector">
              <select
                value={selectedAlgorithm}
                onChange={(e) => setSelectedAlgorithm(e.target.value)}
              >
                <option value="">Auto (train all algorithms)</option>
                {trainingOptions.algorithms.map(algo => (
                  <option key={algo} value={algo}>{algo}</option>
                ))}
              </select>
            </div>

            {selectedAlgorithm && (
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ marginTop: '12px' }}
              >
                <span className="algo-chip">
                  🎯 {selectedAlgorithm}
                </span>
              </motion.div>
            )}
          </div>

          {/* ── Feature Selection ── */}
          <div className="config-section">
            <div className="config-section-header">
              <div className="config-icon" style={{ background: 'var(--gradient-success)', color: 'white' }}>
                <HiViewGrid />
              </div>
              <div>
                <h4>Feature Selection</h4>
                <p>
                  Select which dataset columns to use for training
                  &nbsp;·&nbsp; Target: <strong style={{ color: 'var(--accent-emerald)' }}>
                    {trainingOptions.target_column}
                  </strong>
                </p>
              </div>
            </div>

            {/* Controls bar */}
            <div className="feature-controls">
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                  className="btn-link"
                  onClick={selectAllFeatures}
                  disabled={allFeaturesSelected}
                  id="select-all-features"
                >
                  Select All
                </button>
                <button
                  className="btn-link"
                  onClick={deselectAllFeatures}
                  disabled={selectedFeatures.length === 0}
                  id="deselect-all-features"
                >
                  Deselect All
                </button>
              </div>
              <span className="feature-count-badge">
                ✓ {selectedFeatures.length} / {trainingOptions.features.length} features
              </span>
            </div>

            {/* Search box */}
            {trainingOptions.features.length > 6 && (
              <div className="feature-search-wrapper">
                <HiSearch className="search-icon" />
                <input
                  type="text"
                  className="feature-search"
                  placeholder="Search features..."
                  value={featureSearch}
                  onChange={(e) => setFeatureSearch(e.target.value)}
                  id="feature-search"
                />
              </div>
            )}

            {/* Feature checkbox grid */}
            <div className="feature-grid" id="feature-grid">
              <AnimatePresence>
                {filteredFeatures.map((feature) => {
                  const isChecked = selectedFeatures.includes(feature.name);
                  return (
                    <motion.label
                      key={feature.name}
                      className={`feature-checkbox ${isChecked ? 'checked' : ''}`}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      title={`${feature.name} (${feature.dtype}) · ${feature.unique_values} unique · ${feature.missing_pct}% missing`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleFeature(feature.name)}
                      />
                      <span className="feature-name">{feature.name}</span>
                      <span className="feature-dtype">{feature.dtype}</span>
                    </motion.label>
                  );
                })}
              </AnimatePresence>
            </div>

            {filteredFeatures.length === 0 && featureSearch && (
              <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '20px', fontSize: '0.9rem' }}>
                No features match &quot;{featureSearch}&quot;
              </p>
            )}
          </div>
        </motion.div>
      )}

      {/* Options loading skeleton */}
      {!loading && !result && optionsLoading && (
        <div className="glass-card" style={{ padding: '40px', textAlign: 'center', marginBottom: '24px' }}>
          <div className="spinner" style={{ margin: '0 auto 16px' }} />
          <p style={{ color: 'var(--text-muted)' }}>Loading training options...</p>
        </div>
      )}

      {/* Train Button / Progress / Complete */}
      <motion.div
        className="glass-card"
        style={{ padding: '40px', textAlign: 'center', marginBottom: '24px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {!loading && !result && (
          <>
            <p style={{ fontSize: '3rem', marginBottom: '16px' }}>🧠</p>
            <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '1.2rem' }}>
              Ready to Train
            </h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '24px', maxWidth: '560px', margin: '0 auto 24px' }}>
              {selectedAlgorithm
                ? `Train "${selectedAlgorithm}" with ${selectedFeatures.length} selected feature${selectedFeatures.length !== 1 ? 's' : ''}, hyperparameter tuning (GridSearchCV), and 5-Fold Cross Validation.`
                : `Auto-train all algorithms with ${selectedFeatures.length} selected feature${selectedFeatures.length !== 1 ? 's' : ''}, hyperparameter tuning, 5-Fold CV, and SHAP explanations.`
              }
            </p>

            {/* Summary chips */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '24px', flexWrap: 'wrap' }}>
              {selectedAlgorithm && (
                <span className="algo-chip">🎯 {selectedAlgorithm}</span>
              )}
              {!selectedAlgorithm && (
                <span style={{
                  padding: '6px 14px', borderRadius: '9999px', fontSize: '0.85rem', fontWeight: 600,
                  background: 'rgba(167, 139, 250, 0.12)', color: 'var(--accent-violet)',
                  border: '1px solid rgba(167, 139, 250, 0.25)'
                }}>
                  🔄 Auto (All Algorithms)
                </span>
              )}
              <span className="feature-count-badge">
                📊 {selectedFeatures.length} feature{selectedFeatures.length !== 1 ? 's' : ''}
              </span>
              {trainingOptions?.problem_type && (
                <span className={`badge badge-${trainingOptions.problem_type}`}>
                  {trainingOptions.problem_type}
                </span>
              )}
            </div>

            <button
              className="btn-primary"
              onClick={handleTrain}
              style={{ fontSize: '1rem', padding: '14px 40px' }}
              disabled={trainingOptions?.features && selectedFeatures.length === 0}
              id="start-training-btn"
            >
              <HiBeaker /> Start Training
            </button>
          </>
        )}

        {loading && (
          <>
            <div className="spinner pulse-glow" style={{ margin: '0 auto 20px', width: '60px', height: '60px', borderWidth: '4px' }} />
            <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Training in Progress...</h3>
            <p style={{ color: 'var(--accent-indigo)', fontWeight: 600, fontSize: '1.2rem', marginBottom: '8px' }}>
              {progressText}
            </p>
            <p style={{ color: 'var(--text-muted)' }}>
              {elapsed}s elapsed
            </p>
            {selectedAlgorithm && (
              <div style={{ marginTop: '12px' }}>
                <span className="algo-chip">🎯 {selectedAlgorithm}</span>
              </div>
            )}
          </>
        )}

        {!loading && result && (
          <>
            <HiCheckCircle style={{ fontSize: '3rem', color: 'var(--accent-emerald)', marginBottom: '12px' }} />
            <h3 style={{ fontWeight: 700, marginBottom: '8px', color: 'var(--accent-emerald)' }}>
              Training Complete!
            </h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '16px' }}>
              {result.models?.length === 1
                ? `${result.best_model} trained successfully. View results in the Evaluation page.`
                : `All ${result.models?.length} models trained successfully. View results in the Evaluation page.`
              }
            </p>
            <button className="btn-primary" onClick={() => { setResult(null); }}>
              <HiBeaker /> Configure New Training
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
                  {/* Key metrics inline */}
                  {model.metrics?.accuracy != null && (
                    <p style={{ marginTop: '4px' }}>
                      Accuracy: <strong style={{ color: 'var(--accent-emerald)' }}>
                        {(model.metrics.accuracy * 100).toFixed(2)}%
                      </strong>
                    </p>
                  )}
                  {model.metrics?.f1_score != null && (
                    <p style={{ marginTop: '2px' }}>
                      F1 Score: <strong style={{ color: 'var(--accent-indigo)' }}>
                        {model.metrics.f1_score.toFixed(4)}
                      </strong>
                    </p>
                  )}
                  {model.metrics?.r2_score != null && (
                    <p style={{ marginTop: '2px' }}>
                      R² Score: <strong style={{ color: 'var(--accent-indigo)' }}>
                        {model.metrics.r2_score.toFixed(4)}
                      </strong>
                    </p>
                  )}
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
