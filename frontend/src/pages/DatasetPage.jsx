import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';
import { HiUpload, HiDatabase, HiArrowRight } from 'react-icons/hi';
import { uploadDataset } from '../services/api.js';

export default function DatasetPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);
  const navigate = useNavigate();

  const handleUpload = async () => {
    if (!file) {
      toast.error('Please select a CSV file.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const { data } = await uploadDataset(formData);
      setResult(data);
      toast.success('Dataset uploaded! Now analyze columns to find the best target.');
    } catch (err) {
      const msg = err.response?.data?.detail || 'Upload failed.';
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.name.endsWith('.csv')) {
      setFile(droppedFile);
    } else {
      toast.error('Only CSV files are supported.');
    }
  };

  const summary = result?.summary;

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="section-header">
          <div className="icon gradient-accent" style={{ color: 'white' }}>
            <HiDatabase />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Dataset Upload</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Upload a labeled CSV dataset — the system will intelligently analyze all columns
            </p>
          </div>
        </div>
      </motion.div>

      {/* Upload Zone */}
      <motion.div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        style={{ marginBottom: '24px' }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => setFile(e.target.files[0])}
        />
        <HiUpload style={{ fontSize: '3rem', color: 'var(--accent-indigo)', marginBottom: '16px' }} />
        {file ? (
          <div>
            <p style={{ fontWeight: 600, fontSize: '1.1rem' }}>{file.name}</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              {(file.size / 1024).toFixed(1)} KB — Click or drop to replace
            </p>
          </div>
        ) : (
          <div>
            <p style={{ fontWeight: 600, fontSize: '1.1rem' }}>Drop your CSV here or click to browse</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Supports .csv files</p>
          </div>
        )}
      </motion.div>

      {/* Upload Button */}
      <motion.div
        className="glass-card"
        style={{ padding: '24px', marginBottom: '24px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: '0.95rem', fontWeight: 600 }}>
              No need to specify a target column!
            </p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
              Our AI analyzer will examine every column and recommend the best target candidates.
            </p>
          </div>
          <button
            className="btn-primary"
            onClick={handleUpload}
            disabled={loading || !file}
            style={{ height: '44px' }}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }} />
                Processing...
              </>
            ) : (
              <>
                <HiUpload /> Upload Dataset
              </>
            )}
          </button>
        </div>
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          {/* Summary Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
            <div className="glass-card stat-card">
              <div className="stat-value gradient-text">{summary.num_rows?.toLocaleString()}</div>
              <div className="stat-label">Rows</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-value gradient-text">{summary.num_cols}</div>
              <div className="stat-label">Columns</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-value" style={{ color: summary.total_missing > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
                {summary.total_missing}
              </div>
              <div className="stat-label">Missing Values</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-value gradient-text">{summary.numeric_columns?.length}</div>
              <div className="stat-label">Numeric Cols</div>
            </div>
          </div>

          {/* Column Info */}
          <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
            <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>Column Details</h3>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Data Type</th>
                    <th>Missing</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.columns?.map((col) => (
                    <tr key={col}>
                      <td style={{ fontWeight: 500 }}>{col}</td>
                      <td>
                        <code style={{ fontSize: '0.8rem', background: 'var(--bg-primary)', padding: '2px 8px', borderRadius: '4px' }}>
                          {summary.dtypes[col]}
                        </code>
                      </td>
                      <td style={{ color: summary.missing_values[col] > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
                        {summary.missing_values[col]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Preview */}
          <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
            <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>Data Preview (First 10 Rows)</h3>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    {summary.columns?.map((col) => (
                      <th key={col} style={{ whiteSpace: 'nowrap' }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preview?.map((row, i) => (
                    <tr key={i}>
                      {summary.columns.map((col) => (
                        <td key={col} style={{ whiteSpace: 'nowrap', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Next Step */}
          <motion.div
            className="glass-card"
            style={{
              padding: '32px',
              textAlign: 'center',
              border: '1px solid var(--accent-indigo)',
              background: 'rgba(99, 102, 241, 0.05)',
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <p style={{ fontSize: '2rem', marginBottom: '12px' }}>🧠</p>
            <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Dataset Ready for Analysis</h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '20px', maxWidth: '500px', margin: '0 auto 20px' }}>
              Proceed to the Dataset Analyzer to discover the best target column candidates using AI-powered analysis.
            </p>
            <button
              className="btn-primary"
              onClick={() => navigate('/analyzer')}
              style={{ fontSize: '1rem', padding: '14px 36px' }}
            >
              Analyze Columns <HiArrowRight />
            </button>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}
