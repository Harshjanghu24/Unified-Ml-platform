import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiDocumentReport } from 'react-icons/hi';
import { getMetrics, generateReport } from '../services/api.js';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar
} from 'recharts';

export default function EvaluationPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [reportLoading, setReportLoading] = useState(false);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const { data: res } = await getMetrics();
      setData(res);
    } catch {
      toast.error('No metrics available. Train models first.');
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    setReportLoading(true);
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
      toast.error('Report generation failed.');
    } finally {
      setReportLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '80px' }}>
        <div className="spinner" style={{ margin: '0 auto 16px' }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading metrics...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="glass-card" style={{ padding: '60px', textAlign: 'center', maxWidth: '600px', margin: '40px auto' }}>
        <p style={{ fontSize: '3rem', marginBottom: '16px' }}>📊</p>
        <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>No Evaluation Data</h3>
        <p style={{ color: 'var(--text-muted)' }}>Train models first to see evaluation metrics.</p>
      </div>
    );
  }

  const { models, eval_plots, shap_results, problem_type, best_model } = data;

  // Prepare chart data
  const chartData = models?.map((m) => ({
    name: m.model_name.replace('Classifier', '').replace('Regressor', ''),
    ...Object.fromEntries(
      Object.entries(m.metrics || {}).filter(([k]) =>
        !['confusion_matrix', 'classification_report', 'y_pred', 'y_test'].includes(k)
      ).map(([k, v]) => [k, typeof v === 'number' ? v : 0])
    ),
    training_time: m.training_time,
  }));

  const metricKeys = models?.[0]?.metrics
    ? Object.keys(models[0].metrics).filter(k =>
        !['confusion_matrix', 'classification_report', 'y_pred', 'y_test'].includes(k) &&
        typeof models[0].metrics[k] === 'number'
      )
    : [];

  const colors = ['#818cf8', '#a78bfa', '#22d3ee', '#34d399', '#fbbf24', '#fb7185'];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px', marginBottom: '24px' }}>
          <div className="section-header" style={{ marginBottom: 0 }}>
            <div className="icon gradient-accent" style={{ color: 'white' }}>
              <HiDocumentReport />
            </div>
            <div>
              <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Model Evaluation</h1>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Compare model performance, metrics, and explanations
              </p>
            </div>
          </div>
          <button
            className="btn-primary"
            onClick={downloadReport}
            disabled={reportLoading}
          >
            {reportLoading ? '⏳ Generating...' : '📄 Download PDF Report'}
          </button>
        </div>
      </motion.div>

      {/* Model Comparison Table */}
      <motion.div
        className="glass-card"
        style={{ padding: '24px', marginBottom: '24px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>Model Comparison</h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Time (s)</th>
                {metricKeys.map((k) => (
                  <th key={k}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</th>
                ))}
                <th>Best</th>
              </tr>
            </thead>
            <tbody>
              {models?.map((m, i) => (
                <tr key={i} style={{ background: m.is_best ? 'rgba(251, 191, 36, 0.05)' : undefined }}>
                  <td style={{ fontWeight: 600 }}>{m.model_name}</td>
                  <td>{m.training_time}</td>
                  {metricKeys.map((k) => (
                    <td key={k} style={{ fontWeight: m.is_best ? 700 : 400 }}>
                      {typeof m.metrics[k] === 'number' ? m.metrics[k].toFixed(4) : '-'}
                    </td>
                  ))}
                  <td>{m.is_best ? <span className="badge badge-best">⭐ Best</span> : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Metrics Bar Chart */}
      {chartData && chartData.length > 0 && (
        <motion.div
          className="glass-card"
          style={{ padding: '24px', marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>Metrics Comparison</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={chartData} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f1f5f9',
                }}
              />
              <Legend />
              {metricKeys.map((key, i) => (
                <Bar
                  key={key}
                  dataKey={key}
                  fill={colors[i % colors.length]}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Cross Validation Scores */}
      {models?.some(m => m.cv_scores && Object.keys(m.cv_scores).length > 0) && (
        <motion.div
          className="glass-card"
          style={{ padding: '24px', marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
        >
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>📋 5-Fold Cross Validation</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
            {models.map((m, i) => (
              m.cv_scores && Object.keys(m.cv_scores).length > 0 && (
                <div key={i} style={{ padding: '16px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
                  <h4 style={{ fontWeight: 600, marginBottom: '12px', fontSize: '0.95rem' }}>
                    {m.model_name} {m.is_best ? '⭐' : ''}
                  </h4>
                  {Object.entries(m.cv_scores).map(([metric, scores]) => (
                    <div key={metric} style={{ marginBottom: '8px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                        <span style={{ color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                          {metric.replace(/_/g, ' ')}
                        </span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-indigo)' }}>
                          {scores.mean} ± {scores.std}
                        </span>
                      </div>
                      <div style={{
                        display: 'flex', gap: '4px', marginTop: '4px',
                      }}>
                        {scores.scores?.map((s, j) => (
                          <div
                            key={j}
                            style={{
                              flex: 1, height: '4px', borderRadius: '2px',
                              background: `rgba(129, 140, 248, ${0.3 + (s * 0.7)})`,
                            }}
                            title={`Fold ${j + 1}: ${s}`}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )
            ))}
          </div>
        </motion.div>
      )}

      {/* Evaluation Plots */}
      {eval_plots && Object.keys(eval_plots).length > 0 && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>📈 Evaluation Visualizations</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px', marginBottom: '24px' }}>
            {Object.entries(eval_plots).map(([key, b64]) => (
              b64 && (
                <div key={key} className="glass-card" style={{ padding: '20px' }}>
                  <h4 style={{ fontWeight: 600, marginBottom: '12px', fontSize: '0.95rem' }}>
                    {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </h4>
                  <img src={`data:image/png;base64,${b64}`} alt={key} className="plot-image" />
                </div>
              )
            ))}
          </div>
        </motion.div>
      )}

      {/* SHAP Explainability */}
      {shap_results && (shap_results.summary_plot || shap_results.bar_plot) && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>🔍 SHAP Model Explainability</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px', marginBottom: '24px' }}>
            {shap_results.bar_plot && (
              <div className="glass-card" style={{ padding: '20px' }}>
                <h4 style={{ fontWeight: 600, marginBottom: '12px' }}>SHAP Feature Importance</h4>
                <img src={`data:image/png;base64,${shap_results.bar_plot}`} alt="SHAP Bar" className="plot-image" />
              </div>
            )}
            {shap_results.summary_plot && (
              <div className="glass-card" style={{ padding: '20px' }}>
                <h4 style={{ fontWeight: 600, marginBottom: '12px' }}>SHAP Summary Plot</h4>
                <img src={`data:image/png;base64,${shap_results.summary_plot}`} alt="SHAP Summary" className="plot-image" />
              </div>
            )}
          </div>

          {/* SHAP Feature Table */}
          {shap_results.feature_importance && shap_results.feature_importance.length > 0 && (
            <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
              <h4 style={{ fontWeight: 600, marginBottom: '12px' }}>Top Features by SHAP Value</h4>
              <div style={{ display: 'grid', gap: '6px' }}>
                {shap_results.feature_importance.slice(0, 10).map((f, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '6px 0' }}>
                    <span style={{ width: '24px', fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      #{i + 1}
                    </span>
                    <span style={{ flex: 1, fontSize: '0.9rem' }}>{f.feature}</span>
                    <div style={{
                      width: `${Math.max(f.mean_shap_value * 600, 8)}px`,
                      height: '8px',
                      background: 'var(--gradient-accent)',
                      borderRadius: '4px',
                      maxWidth: '200px',
                    }} />
                    <span style={{ fontWeight: 600, fontSize: '0.85rem', minWidth: '60px', textAlign: 'right' }}>
                      {f.mean_shap_value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
