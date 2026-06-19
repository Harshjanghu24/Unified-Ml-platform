import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';
import {
  HiCog, HiDatabase, HiTable, HiCheckCircle, HiSearch, HiArrowRight,
  HiExclamationCircle, HiLightningBolt
} from 'react-icons/hi';
import {
  getDatasetSummary,
  preprocessAuto,
  preprocessManual,
  getPreprocessingReport,
  getEncodingRecommendations
} from '../services/api.js';

export default function PreprocessingPage() {
  const navigate = useNavigate();
  
  // Loading states
  const [loadingDataset, setLoadingDataset] = useState(true);
  const [processing, setProcessing] = useState(false);
  
  // Dataset information
  const [datasetInfo, setDatasetInfo] = useState(null); // original summary & preview
  const [targetColumn, setTargetColumn] = useState('');
  
  // Mode selection
  const [mode, setMode] = useState('auto'); // 'auto' | 'manual'
  
  // Recommendations & Manual Encoding States
  const [recommendations, setRecommendations] = useState([]);
  const [columnEncodings, setColumnEncodings] = useState({});
  const [encodingSearch, setEncodingSearch] = useState('');
  
  // General Manual Configuration States
  const [missingMethod, setMissingMethod] = useState('median');
  const [duplicateHandling, setDuplicateHandling] = useState('remove');
  const [scalingMethod, setScalingMethod] = useState('standard');
  const [outlierMethod, setOutlierMethod] = useState('none');
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [featureSearch, setFeatureSearch] = useState('');
  
  // Results / Reports / Aborts
  const [report, setReport] = useState(null);
  const [previewTab, setPreviewTab] = useState('original'); // 'original' | 'processed'
  const [abortError, setAbortError] = useState(null);

  // Fetch dataset info on load
  useEffect(() => {
    fetchDatasetInfo();
  }, []);

  const fetchDatasetInfo = async () => {
    setLoadingDataset(true);
    setAbortError(null);
    try {
      const { data } = await getDatasetSummary();
      setDatasetInfo(data);
      if (data.summary) {
        const cols = data.summary.columns || [];
        const target = data.summary.target_info?.name || '';
        setTargetColumn(target);
        
        const featureCols = cols.filter(c => c !== target);
        setSelectedFeatures(featureCols);
      }
      
      // Load cardinality recommendations for manual mode
      try {
        const recsRes = await getEncodingRecommendations();
        setRecommendations(recsRes.data || []);
        
        // Populate initial recommendations state
        const initialEncodings = {};
        recsRes.data.forEach(r => {
          initialEncodings[r.column] = r.recommendation;
        });
        setColumnEncodings(initialEncodings);
      } catch (e) {
        console.error("Failed to load recommendations", e);
      }
      
      // Try to load any existing preprocessing report
      try {
        const reportRes = await getPreprocessingReport();
        if (reportRes.data) {
          setReport(reportRes.data);
          setPreviewTab('processed');
        }
      } catch {
        // No existing report
      }
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to load dataset details.');
    } finally {
      setLoadingDataset(false);
    }
  };

  const handleRunAuto = async () => {
    setProcessing(true);
    setAbortError(null);
    try {
      const { data } = await preprocessAuto();
      setReport(data);
      setPreviewTab('processed');
      toast.success('Automatic Preprocessing Completed!');
    } catch (err) {
      const errMsg = err.response?.data?.detail || 'Auto preprocessing failed.';
      if (err.response?.status === 400 && errMsg.includes('Encoding would generate')) {
        setAbortError(errMsg);
        toast.error('OHE Aborted to Prevent Crash');
      } else {
        toast.error(errMsg);
      }
    } finally {
      setProcessing(false);
    }
  };

  const handleRunManual = async () => {
    if (selectedFeatures.length === 0) {
      toast.error('Please select at least one feature.');
      return;
    }
    setProcessing(true);
    setAbortError(null);
    try {
      const config = {
        missing_value_method: missingMethod,
        duplicate_handling: duplicateHandling,
        column_encodings: columnEncodings,
        scaling_method: scalingMethod,
        outlier_method: outlierMethod,
        selected_features: selectedFeatures,
      };
      const { data } = await preprocessManual(config);
      setReport(data);
      setPreviewTab('processed');
      toast.success('Manual Preprocessing Applied!');
    } catch (err) {
      const errMsg = err.response?.data?.detail || 'Manual preprocessing failed.';
      if (err.response?.status === 400 && errMsg.includes('Encoding would generate')) {
        setAbortError(errMsg);
        toast.error('OHE Aborted to Prevent Crash');
      } else {
        toast.error(errMsg);
      }
    } finally {
      setProcessing(false);
    }
  };

  const handleSelectAllFeatures = () => {
    if (!datasetInfo) return;
    const cols = datasetInfo.summary.columns || [];
    setSelectedFeatures(cols.filter(c => c !== targetColumn));
  };

  const handleDeselectAllFeatures = () => {
    setSelectedFeatures([]);
  };

  const handleToggleFeature = (col) => {
    setSelectedFeatures(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  const handleSetColumnStrategy = (col, strategy) => {
    setColumnEncodings(prev => ({
      ...prev,
      [col]: strategy
    }));
  };

  if (loadingDataset) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <span className="spinner" style={{ width: '40px', height: '40px', borderWidth: '3px' }} />
      </div>
    );
  }

  if (!datasetInfo) {
    return (
      <div className="glass-card" style={{ padding: '40px', textAlign: 'center', maxWidth: '600px', margin: '40px auto' }}>
        <HiDatabase style={{ fontSize: '3rem', color: 'var(--text-muted)', marginBottom: '16px' }} />
        <h3 style={{ fontWeight: 700, marginBottom: '12px' }}>No Dataset Uploaded</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>
          Upload a dataset to configure the adaptive preprocessing options.
        </p>
        <button className="btn-primary" onClick={() => navigate('/dataset')}>
          Go to Upload
        </button>
      </div>
    );
  }

  const { summary } = datasetInfo;
  const targetCol = targetColumn || summary.target_info?.name;

  const filteredColumns = (summary.columns || []).filter(c => 
    c !== targetCol && c.toLowerCase().includes(featureSearch.toLowerCase())
  );

  const filteredRecommendations = recommendations.filter(r => 
    r.column.toLowerCase().includes(encodingSearch.toLowerCase())
  );

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', paddingBottom: '60px' }}>
      
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="section-header">
          <div className="icon gradient-accent" style={{ color: 'white' }}>
            <HiCog />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Dataset Preprocessing</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Clean, scale, and transform your data. Featuring an adaptive encoding engine to guard against memory crashes.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Dataset Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '16px', marginBottom: '24px' }}>
        <div className="glass-card stat-card">
          <div className="stat-value gradient-text">{summary.num_rows?.toLocaleString()}</div>
          <div className="stat-label">Total Rows</div>
        </div>
        <div className="glass-card stat-card">
          <div className="stat-value gradient-text">{summary.num_cols}</div>
          <div className="stat-label">Total Columns</div>
        </div>
        <div className="glass-card stat-card">
          <div className="stat-value" style={{ color: summary.total_missing > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
            {summary.total_missing}
          </div>
          <div className="stat-label">Missing Values</div>
        </div>
        <div className="glass-card stat-card">
          <div className="stat-value" style={{ color: 'var(--accent-amber)' }}>
            {summary.duplicates || 0}
          </div>
          <div className="stat-label">Duplicates</div>
        </div>
        <div className="glass-card stat-card">
          <div className="stat-value gradient-text">
            {summary.memory_usage_mb > 1024 
              ? `${(summary.memory_usage_mb / 1024).toFixed(1)} GB`
              : `${summary.memory_usage_mb} MB`}
          </div>
          <div className="stat-label">Memory Size</div>
        </div>
      </div>

      {/* Mode Selection */}
      <div className="glass-card" style={{ padding: '8px', marginBottom: '24px', display: 'inline-flex', gap: '4px', borderRadius: '12px' }}>
        <button
          className={mode === 'auto' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setMode('auto')}
          style={{ padding: '8px 24px', border: 'none', borderRadius: '8px', fontWeight: 600 }}
        >
          Automatic Mode
        </button>
        <button
          className={mode === 'manual' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setMode('manual')}
          style={{ padding: '8px 24px', border: 'none', borderRadius: '8px', fontWeight: 600 }}
        >
          Manual Mode
        </button>
      </div>

      {/* Main Options Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: mode === 'manual' ? '1fr 320px' : '1fr', gap: '24px', marginBottom: '32px' }}>
        
        {/* Options Panel */}
        <div className="glass-card" style={{ padding: '28px' }}>
          {mode === 'auto' ? (
            <div>
              <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Guided One-Click Preprocessing</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '24px' }}>
                Leverages adaptive cardinality thresholds to dynamically choose OHE or Label encoding. Drops high-cardinality/identifier columns.
              </p>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '28px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.95rem' }}>
                  <HiCheckCircle style={{ color: 'var(--accent-emerald)', fontSize: '1.3rem' }} />
                  <div>
                    <span style={{ fontWeight: 600 }}>Missing Value Imputation:</span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: '6px' }}>Median for numeric, Mode for categorical.</span>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.95rem' }}>
                  <HiCheckCircle style={{ color: 'var(--accent-emerald)', fontSize: '1.3rem' }} />
                  <div>
                    <span style={{ fontWeight: 600 }}>Duplicate Handling:</span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: '6px' }}>Removes identical rows.</span>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.95rem' }}>
                  <HiCheckCircle style={{ color: 'var(--accent-emerald)', fontSize: '1.3rem' }} />
                  <div>
                    <span style={{ fontWeight: 600 }}>Adaptive Encoding Strategy:</span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: '6px' }}>One-Hot (≤20 unique values), Label encoding (21 - 1000), Drop column (&gt;1000).</span>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.95rem' }}>
                  <HiCheckCircle style={{ color: 'var(--accent-emerald)', fontSize: '1.3rem' }} />
                  <div>
                    <span style={{ fontWeight: 600 }}>Outlier & Scaling:</span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: '6px' }}>StandardScaler scaling, outliers retained by default.</span>
                  </div>
                </div>
              </div>

              <button
                className="btn-primary"
                onClick={handleRunAuto}
                disabled={processing}
                style={{ padding: '12px 32px', fontSize: '1rem' }}
              >
                {processing ? (
                  <>
                    <span className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }} />
                    Running Auto Preprocessing...
                  </>
                ) : 'Run Auto Preprocessing'}
              </button>
            </div>
          ) : (
            <div>
              <h3 style={{ fontWeight: 700, marginBottom: '20px' }}>Custom Preprocessing Pipeline</h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
                <div>
                  <label style={{ display: 'block', fontWeight: 600, marginBottom: '8px', fontSize: '0.9rem' }}>
                    Missing Values Imputation
                  </label>
                  <select
                    value={missingMethod}
                    onChange={(e) => setMissingMethod(e.target.value)}
                    style={{
                      width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)',
                      background: 'var(--bg-primary)', color: 'var(--text-primary)', outline: 'none'
                    }}
                  >
                    <option value="mean">Mean Imputation</option>
                    <option value="median">Median Imputation</option>
                    <option value="mode">Mode Imputation</option>
                    <option value="drop_rows">Drop Rows containing NaNs</option>
                    <option value="drop_cols">Drop Columns containing NaNs</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontWeight: 600, marginBottom: '8px', fontSize: '0.9rem' }}>
                    Duplicate Handling
                  </label>
                  <select
                    value={duplicateHandling}
                    onChange={(e) => setDuplicateHandling(e.target.value)}
                    style={{
                      width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)',
                      background: 'var(--bg-primary)', color: 'var(--text-primary)', outline: 'none'
                    }}
                  >
                    <option value="remove">Remove Duplicates</option>
                    <option value="keep">Keep Duplicates</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontWeight: 600, marginBottom: '8px', fontSize: '0.9rem' }}>
                    Feature Scaling
                  </label>
                  <select
                    value={scalingMethod}
                    onChange={(e) => setScalingMethod(e.target.value)}
                    style={{
                      width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)',
                      background: 'var(--bg-primary)', color: 'var(--text-primary)', outline: 'none'
                    }}
                  >
                    <option value="standard">StandardScaler (Z-score)</option>
                    <option value="minmax">MinMaxScaler (0 - 1)</option>
                    <option value="robust">RobustScaler (IQR based)</option>
                    <option value="none">None (No Scaling)</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontWeight: 600, marginBottom: '8px', fontSize: '0.9rem' }}>
                    Outlier Treatment
                  </label>
                  <select
                    value={outlierMethod}
                    onChange={(e) => setOutlierMethod(e.target.value)}
                    style={{
                      width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)',
                      background: 'var(--bg-primary)', color: 'var(--text-primary)', outline: 'none'
                    }}
                  >
                    <option value="none">None (Keep Outliers)</option>
                    <option value="iqr">IQR Method (Clip outliers)</option>
                    <option value="zscore">Z-Score Method (Clip outliers)</option>
                  </select>
                </div>
              </div>

              {/* Categorical Encoding Strategy Customizer */}
              <div style={{ marginBottom: '32px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h4 style={{ fontWeight: 700, margin: 0 }}>Categorical Encoders Strategy Override</h4>
                  <div style={{ position: 'relative', width: '220px' }}>
                    <input
                      type="text"
                      placeholder="Search categories..."
                      value={encodingSearch}
                      onChange={(e) => setEncodingSearch(e.target.value)}
                      style={{
                        width: '100%', padding: '6px 12px 6px 28px', borderRadius: '6px',
                        border: '1px solid var(--border-color)', background: 'var(--bg-primary)',
                        color: 'var(--text-primary)', fontSize: '0.8rem', outline: 'none'
                      }}
                    />
                    <HiSearch style={{ position: 'absolute', left: '8px', top: '8px', color: 'var(--text-muted)' }} />
                  </div>
                </div>

                {recommendations.length > 0 ? (
                  <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', maxHeight: '280px', overflowY: 'auto' }}>
                    <table className="data-table" style={{ margin: 0, fontSize: '0.85rem' }}>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Unique Values</th>
                          <th>Recommendation</th>
                          <th>Encoding Override</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredRecommendations.map(rec => {
                          const val = columnEncodings[rec.column] || rec.recommendation;
                          return (
                            <tr key={rec.column}>
                              <td style={{ fontWeight: 600 }}>{rec.column}</td>
                              <td>{rec.nunique.toLocaleString()}</td>
                              <td>
                                <span style={{
                                  padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600,
                                  background: rec.is_identifier || rec.nunique > 1000 
                                    ? 'rgba(244,63,94,0.1)' 
                                    : (rec.nunique > 20 ? 'rgba(245,158,11,0.1)' : 'rgba(16,185,129,0.1)'),
                                  color: rec.is_identifier || rec.nunique > 1000 
                                    ? 'var(--accent-rose)' 
                                    : (rec.nunique > 20 ? 'var(--accent-amber)' : 'var(--accent-emerald)')
                                }}>
                                  {rec.is_identifier 
                                    ? 'Drop (Identifier)' 
                                    : `${rec.cardinality_level} Card. (${rec.recommendation.toUpperCase()})`}
                                </span>
                              </td>
                              <td>
                                <div style={{ display: 'flex', gap: '8px' }}>
                                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                                    <input
                                      type="radio"
                                      name={`enc-${rec.column}`}
                                      checked={val === 'onehot'}
                                      onChange={() => handleSetColumnStrategy(rec.column, 'onehot')}
                                    />
                                    One-Hot
                                  </label>
                                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                                    <input
                                      type="radio"
                                      name={`enc-${rec.column}`}
                                      checked={val === 'label'}
                                      onChange={() => handleSetColumnStrategy(rec.column, 'label')}
                                    />
                                    Label
                                  </label>
                                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                                    <input
                                      type="radio"
                                      name={`enc-${rec.column}`}
                                      checked={val === 'drop'}
                                      onChange={() => handleSetColumnStrategy(rec.column, 'drop')}
                                    />
                                    Drop
                                  </label>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)', background: 'var(--bg-primary)', borderRadius: '8px' }}>
                    No categorical columns found in the dataset.
                  </div>
                )}
              </div>

              <button
                className="btn-primary"
                onClick={handleRunManual}
                disabled={processing}
                style={{ padding: '12px 32px', fontSize: '1rem' }}
              >
                {processing ? (
                  <>
                    <span className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }} />
                    Applying Preprocessing...
                  </>
                ) : 'Apply Manual Preprocessing'}
              </button>
            </div>
          )}
        </div>

        {/* Right Sidebar - Feature Selector */}
        {mode === 'manual' && (
          <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', height: '100%' }}>
            <h4 style={{ fontWeight: 700, marginBottom: '12px' }}>Feature Columns</h4>
            
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <button className="btn-secondary" onClick={handleSelectAllFeatures} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>
                Select All
              </button>
              <button className="btn-secondary" onClick={handleDeselectAllFeatures} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>
                Clear All
              </button>
            </div>

            <div style={{ position: 'relative', marginBottom: '12px' }}>
              <input
                type="text"
                placeholder="Search features..."
                value={featureSearch}
                onChange={(e) => setFeatureSearch(e.target.value)}
                style={{
                  width: '100%', padding: '8px 12px 8px 32px', borderRadius: '6px',
                  border: '1px solid var(--border-color)', background: 'var(--bg-primary)',
                  color: 'var(--text-primary)', fontSize: '0.85rem', outline: 'none'
                }}
              />
              <HiSearch style={{ position: 'absolute', left: '10px', top: '10px', color: 'var(--text-muted)' }} />
            </div>

            <div style={{ flex: 1, overflowY: 'auto', maxHeight: '380px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {filteredColumns.map(col => (
                <label
                  key={col}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px',
                    borderRadius: '6px', background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                    cursor: 'pointer', fontSize: '0.85rem'
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedFeatures.includes(col)}
                    onChange={() => handleToggleFeature(col)}
                  />
                  <span style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {col}
                  </span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Dimensional Explosion Abort Handler Alert */}
      <AnimatePresence>
        {abortError && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            style={{
              background: 'rgba(244, 63, 94, 0.08)', border: '1px solid var(--accent-rose)',
              borderRadius: '12px', padding: '24px', marginBottom: '24px', display: 'flex', gap: '16px'
            }}
          >
            <HiExclamationCircle style={{ color: 'var(--accent-rose)', fontSize: '2.5rem', flexShrink: 0 }} />
            <div>
              <h4 style={{ color: 'var(--accent-rose)', fontWeight: 700, marginBottom: '6px', fontSize: '1.05rem' }}>
                Dimensional Explosion Blocked
              </h4>
              <pre style={{
                fontFamily: 'monospace', color: 'var(--text-primary)', fontSize: '0.88rem',
                whiteSpace: 'pre-wrap', background: 'var(--bg-primary)', padding: '12px',
                borderRadius: '8px', border: '1px solid var(--border-color)', marginTop: '8px'
              }}>
                {abortError}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Preprocessing Metrics & Encoding Analysis Dashboard */}
      <AnimatePresence>
        {report && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
            
            {/* Memory Metrics Summary Cards */}
            <div className="glass-card" style={{ padding: '24px', marginBottom: '24px', border: '1px solid var(--accent-indigo)' }}>
              <h3 style={{ fontWeight: 700, marginBottom: '16px', color: 'var(--accent-indigo)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <HiLightningBolt /> Preprocessing & Memory Report
              </h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '16px', marginBottom: '20px' }}>
                <div className="stat-card" style={{ background: 'rgba(99,102,241,0.05)', padding: '16px' }}>
                  <div className="stat-value" style={{ color: 'var(--text-primary)', fontSize: '1.4rem' }}>
                    {report.processing_summary.columns_before_encoding ?? summary.num_cols - 1}
                  </div>
                  <div className="stat-label">Columns Before Encoding</div>
                </div>
                <div className="stat-card" style={{ background: 'rgba(99,102,241,0.05)', padding: '16px' }}>
                  <div className="stat-value" style={{ color: 'var(--text-primary)', fontSize: '1.4rem' }}>
                    {report.processing_summary.columns_after_encoding ?? report.after_stats.num_cols - 1}
                  </div>
                  <div className="stat-label">Columns After Encoding</div>
                </div>
                <div className="stat-card" style={{ background: 'rgba(16,185,129,0.08)', padding: '16px' }}>
                  <div className="stat-value" style={{ color: 'var(--accent-emerald)', fontSize: '1.4rem' }}>
                    {report.processing_summary.estimated_memory_saved_mb 
                      ? `${report.processing_summary.estimated_memory_saved_mb.toLocaleString()} MB` 
                      : '0 MB'}
                  </div>
                  <div className="stat-label">Estimated Memory Saved</div>
                </div>
                <div className="stat-card" style={{ background: 'rgba(244,63,94,0.05)', padding: '16px' }}>
                  <div className="stat-value" style={{ color: 'var(--accent-rose)', fontSize: '1.4rem' }}>
                    {report.processing_summary.high_cardinality_removed_count ?? 0}
                  </div>
                  <div className="stat-label">High-Card. Columns Removed</div>
                </div>
                <div className="stat-card" style={{ background: 'rgba(99,102,241,0.05)', padding: '16px' }}>
                  <div className="stat-value" style={{ color: 'var(--text-primary)', fontSize: '0.9rem', fontWeight: 700, height: '42px', display: 'flex', alignItems: 'center' }}>
                    {report.processing_summary.encoding_strategy_used ?? 'Adaptive'}
                  </div>
                  <div className="stat-label">Encoding Strategy</div>
                </div>
                <div className="stat-card" style={{ background: 'rgba(99,102,241,0.05)', padding: '16px' }}>
                  <div className="stat-value" style={{ color: 'var(--text-primary)', fontSize: '1.4rem' }}>
                    {report.processing_summary.processing_time}s
                  </div>
                  <div className="stat-label">Execution Time</div>
                </div>
              </div>
            </div>

            {/* Before vs After & Encoding Analysis Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
              
              {/* Before vs After Statistics */}
              <div className="glass-card" style={{ padding: '24px' }}>
                <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>Before vs After Statistics</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table" style={{ width: '100%' }}>
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Before</th>
                        <th>After</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ fontWeight: 500 }}>Rows</td>
                        <td>{report.before_stats.num_rows.toLocaleString()}</td>
                        <td>{report.after_stats.num_rows.toLocaleString()}</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 500 }}>Columns</td>
                        <td>{report.before_stats.num_cols}</td>
                        <td>{report.after_stats.num_cols}</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 500 }}>Missing Values</td>
                        <td style={{ color: report.before_stats.missing_values > 0 ? 'var(--accent-rose)' : 'inherit' }}>
                          {report.before_stats.missing_values}
                        </td>
                        <td style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>
                          {report.after_stats.missing_values}
                        </td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 500 }}>Duplicates</td>
                        <td style={{ color: report.before_stats.duplicates > 0 ? 'var(--accent-amber)' : 'inherit' }}>
                          {report.before_stats.duplicates}
                        </td>
                        <td style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>
                          {report.after_stats.duplicates}
                        </td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 500 }}>Memory Usage</td>
                        <td>{report.before_stats.memory_usage_mb} MB</td>
                        <td>{report.after_stats.memory_usage_mb} MB</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Encoding Analysis table */}
              <div className="glass-card" style={{ padding: '24px' }}>
                <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>Encoding Analysis Summary</h3>
                {report.encoding_analysis && report.encoding_analysis.length > 0 ? (
                  <div style={{ overflowY: 'auto', maxHeight: '240px' }}>
                    <table className="data-table" style={{ width: '100%', fontSize: '0.85rem', margin: 0 }}>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Unique Values</th>
                          <th>Strategy Applied</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.encoding_analysis.map(colAnalysis => (
                          <tr key={colAnalysis.column}>
                            <td style={{ fontWeight: 600 }}>{colAnalysis.column}</td>
                            <td>{colAnalysis.unique_values.toLocaleString()}</td>
                            <td>
                              <span style={{
                                padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600,
                                background: colAnalysis.encoding === 'One-Hot' 
                                  ? 'rgba(16,185,129,0.1)' 
                                  : (colAnalysis.encoding === 'Label Encoding' ? 'rgba(245,158,11,0.1)' : 'rgba(244,63,94,0.1)'),
                                color: colAnalysis.encoding === 'One-Hot' 
                                  ? 'var(--accent-emerald)' 
                                  : (colAnalysis.encoding === 'Label Encoding' ? 'var(--accent-amber)' : 'var(--accent-rose)')
                              }}>
                                {colAnalysis.encoding}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)' }}>
                    No categorical features encoded in this run.
                  </div>
                )}
              </div>

            </div>

            {/* Preview comparison */}
            <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <HiTable /> Preview Comparison
                </h3>
                <div style={{ display: 'flex', gap: '4px', background: 'var(--bg-primary)', padding: '4px', borderRadius: '8px' }}>
                  <button
                    className={previewTab === 'original' ? 'btn-primary' : 'btn-secondary'}
                    onClick={() => setPreviewTab('original')}
                    style={{ padding: '6px 16px', fontSize: '0.8rem', border: 'none', borderRadius: '6px' }}
                  >
                    Original Dataset
                  </button>
                  <button
                    className={previewTab === 'processed' ? 'btn-primary' : 'btn-secondary'}
                    onClick={() => setPreviewTab('processed')}
                    style={{ padding: '6px 16px', fontSize: '0.8rem', border: 'none', borderRadius: '6px' }}
                  >
                    Processed Dataset
                  </button>
                </div>
              </div>

              <div style={{ overflowX: 'auto', maxHeight: '320px', border: '1px solid var(--border-color)', borderRadius: '8px' }}>
                <table className="data-table" style={{ margin: 0 }}>
                  <thead>
                    <tr>
                      {previewTab === 'original' ? (
                        summary.columns?.map(col => <th key={col}>{col}</th>)
                      ) : (
                        report.processed_preview.length > 0
                          ? Object.keys(report.processed_preview[0]).map(col => <th key={col}>{col}</th>)
                          : <th>No data</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {previewTab === 'original' ? (
                      datasetInfo.preview?.map((row, i) => (
                        <tr key={i}>
                          {summary.columns?.map(col => (
                            <td key={col} style={{ whiteSpace: 'nowrap', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {String(row[col] ?? '')}
                            </td>
                          ))}
                        </tr>
                      ))
                    ) : (
                      report.processed_preview?.map((row, i) => (
                        <tr key={i}>
                          {Object.keys(row).map(col => (
                            <td key={col} style={{ whiteSpace: 'nowrap', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {typeof row[col] === 'number' ? row[col].toFixed(4).replace(/\.?0+$/, '') : String(row[col] ?? '')}
                            </td>
                          ))}
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Proceed prompt */}
            <div className="glass-card" style={{ padding: '32px', textAlign: 'center', border: '1px solid var(--accent-indigo)', background: 'rgba(99,102,241,0.05)' }}>
              <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Dataset Cleaned & Ready</h3>
              <p style={{ color: 'var(--text-muted)', marginBottom: '20px' }}>
                The preprocessed dataset has been stored. You can now proceed to model training.
              </p>
              <button className="btn-primary" onClick={() => navigate('/training')} style={{ fontSize: '1rem', padding: '14px 36px' }}>
                Proceed to Training <HiArrowRight />
              </button>
            </div>

          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
