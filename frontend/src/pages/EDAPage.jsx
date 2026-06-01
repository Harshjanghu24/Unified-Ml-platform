import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiChartBar } from 'react-icons/hi';
import { getEDA } from '../services/api.js';

export default function EDAPage() {
  const [plots, setPlots] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('all');

  const fetchEDA = async () => {
    setLoading(true);
    try {
      const { data } = await getEDA();
      setPlots(data.plots);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to load EDA. Upload a dataset and train first.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEDA();
  }, []);

  const tabs = [
    { key: 'all', label: 'All' },
    { key: 'distribution', label: 'Distributions' },
    { key: 'correlation', label: 'Correlation' },
    { key: 'scatter', label: 'Scatter' },
  ];

  const getFilteredPlots = () => {
    if (!plots) return {};
    if (activeTab === 'all') return plots;
    if (activeTab === 'distribution') return {
      histograms: plots.histograms,
      box_plots: plots.box_plots,
      target_distribution: plots.target_distribution,
    };
    if (activeTab === 'correlation') return {
      correlation_heatmap: plots.correlation_heatmap,
    };
    if (activeTab === 'scatter') return {
      scatter_plots: plots.scatter_plots,
    };
    return plots;
  };

  const plotLabels = {
    histograms: 'Feature Distributions (Histograms)',
    box_plots: 'Box Plots',
    correlation_heatmap: 'Correlation Heatmap',
    target_distribution: 'Target Column Distribution',
    scatter_plots: 'Scatter Plots (Top Correlations)',
  };

  const filtered = getFilteredPlots();

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="section-header">
          <div className="icon gradient-accent" style={{ color: 'white' }}>
            <HiChartBar />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Exploratory Data Analysis</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Visual analytics dashboard for your dataset
            </p>
          </div>
        </div>
      </motion.div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', flexWrap: 'wrap' }}>
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              padding: '8px 20px',
              borderRadius: '8px',
              border: activeTab === tab.key ? '1px solid var(--accent-indigo)' : '1px solid var(--border-color)',
              background: activeTab === tab.key ? 'rgba(99, 102, 241, 0.1)' : 'var(--bg-card)',
              color: activeTab === tab.key ? 'var(--accent-indigo)' : 'var(--text-secondary)',
              fontWeight: activeTab === tab.key ? 600 : 400,
              cursor: 'pointer',
              fontSize: '0.9rem',
              transition: 'all 0.2s ease',
            }}
          >
            {tab.label}
          </button>
        ))}
        <button className="btn-secondary" onClick={fetchEDA} disabled={loading} style={{ marginLeft: 'auto' }}>
          {loading ? 'Loading...' : '🔄 Refresh'}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '60px' }}>
          <div className="spinner" style={{ margin: '0 auto 16px' }} />
          <p style={{ color: 'var(--text-muted)' }}>Generating visualizations...</p>
        </div>
      )}

      {/* No Data */}
      {!loading && !plots && (
        <div className="glass-card" style={{ padding: '60px', textAlign: 'center' }}>
          <p style={{ fontSize: '3rem', marginBottom: '16px' }}>📊</p>
          <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>No EDA Available</h3>
          <p style={{ color: 'var(--text-muted)' }}>
            Upload a dataset and run training to generate visualizations.
          </p>
        </div>
      )}

      {/* Plots Grid */}
      {!loading && plots && (
        <div style={{ display: 'grid', gap: '24px' }}>
          {Object.entries(filtered).map(([key, b64], i) => {
            if (!b64) return null;
            return (
              <motion.div
                key={key}
                className="glass-card"
                style={{ padding: '24px' }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <h3 style={{ fontWeight: 700, marginBottom: '16px', fontSize: '1.1rem' }}>
                  {plotLabels[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </h3>
                <img
                  src={`data:image/png;base64,${b64}`}
                  alt={plotLabels[key] || key}
                  className="plot-image"
                  style={{ maxWidth: '100%' }}
                />
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
