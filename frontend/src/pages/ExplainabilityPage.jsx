import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { HiLightBulb } from 'react-icons/hi';
import { getMetrics } from '../services/api.js';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from 'recharts';

export default function ExplainabilityPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const { data: res } = await getMetrics();
      setData(res);
    } catch {
      // silent — no data yet
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '80px' }}>
        <div className="spinner" style={{ margin: '0 auto 16px' }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading explainability data...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="glass-card" style={{ padding: '60px', textAlign: 'center', maxWidth: '600px', margin: '40px auto' }}>
        <p style={{ fontSize: '3rem', marginBottom: '16px' }}>🔍</p>
        <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>No Explainability Data</h3>
        <p style={{ color: 'var(--text-muted)' }}>
          Train models first to see SHAP explanations and feature importance analysis.
        </p>
      </div>
    );
  }

  const { shap_results, feature_selection, best_model, problem_type } = data;

  // SHAP feature importance bar chart data
  const shapChartData = shap_results?.feature_importance?.slice(0, 15)?.map(f => ({
    feature: f.feature.length > 18 ? f.feature.slice(0, 18) + '…' : f.feature,
    value: f.mean_shap_value,
  })) || [];

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="section-header">
          <div className="icon gradient-accent" style={{ color: 'white' }}>
            <HiLightBulb />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Explainability Dashboard</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              SHAP-based model explanations and feature importance analysis
            </p>
          </div>
        </div>
      </motion.div>

      {/* Best Model Info */}
      <motion.div
        className="glass-card"
        style={{ padding: '24px', marginBottom: '24px', borderLeft: '4px solid var(--accent-indigo)' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ fontWeight: 700, fontSize: '1.1rem' }}>
              ⭐ Best Model: {best_model}
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
              Problem Type: <span className={`badge badge-${problem_type}`} style={{ fontSize: '0.8rem' }}>
                {problem_type === 'binary' ? '🎯 Binary' : problem_type === 'multiclass' ? '🌈 Multi-Class' : '📈 Regression'}
              </span>
            </p>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Explanations generated using SHAP (SHapley Additive exPlanations)
          </div>
        </div>
      </motion.div>

      {/* SHAP Feature Importance Chart */}
      {shapChartData.length > 0 && (
        <motion.div
          className="glass-card"
          style={{ padding: '24px', marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <h3 style={{ fontWeight: 700, marginBottom: '20px' }}>📊 SHAP Feature Importance</h3>
          <ResponsiveContainer width="100%" height={Math.max(300, shapChartData.length * 32)}>
            <BarChart data={shapChartData} layout="vertical" barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" stroke="#94a3b8" fontSize={12} />
              <YAxis type="category" dataKey="feature" stroke="#94a3b8" fontSize={11} width={150} />
              <Tooltip
                contentStyle={{
                  background: '#1e293b', border: '1px solid #334155',
                  borderRadius: '8px', color: '#f1f5f9',
                }}
                formatter={(val) => [`${val.toFixed(4)}`, 'Mean |SHAP|']}
              />
              <Bar dataKey="value" fill="#818cf8" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* SHAP Plots */}
      {shap_results && (shap_results.summary_plot || shap_results.bar_plot) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>🔬 SHAP Visualizations</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px', marginBottom: '24px' }}>
            {shap_results.bar_plot && (
              <div className="glass-card" style={{ padding: '20px' }}>
                <h4 style={{ fontWeight: 600, marginBottom: '12px' }}>SHAP Bar Plot</h4>
                <img src={`data:image/png;base64,${shap_results.bar_plot}`} alt="SHAP Bar" className="plot-image" />
              </div>
            )}
            {shap_results.summary_plot && (
              <div className="glass-card" style={{ padding: '20px' }}>
                <h4 style={{ fontWeight: 600, marginBottom: '12px' }}>SHAP Summary (Beeswarm)</h4>
                <img src={`data:image/png;base64,${shap_results.summary_plot}`} alt="SHAP Summary" className="plot-image" />
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Feature Selection Results */}
      {feature_selection && !feature_selection.error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
        >
          <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>🎯 Feature Selection Methods</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '16px', marginBottom: '24px' }}>
            {/* Random Forest Importance */}
            {feature_selection.random_forest_importance && !feature_selection.random_forest_importance.error && (
              <div className="glass-card" style={{ padding: '24px' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '16px', color: 'var(--accent-indigo)' }}>
                  🌲 Random Forest Importance
                </h4>
                {feature_selection.random_forest_importance.slice(0, 10).map((f, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-secondary)', flex: 1 }}>{f.feature}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{
                        width: `${Math.max(f.importance * 250, 4)}px`, height: '6px',
                        background: 'var(--gradient-accent)', borderRadius: '3px',
                      }} />
                      <span style={{ fontWeight: 600, minWidth: '50px', textAlign: 'right' }}>{f.importance}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Mutual Information */}
            {feature_selection.mutual_information && !feature_selection.mutual_information.error && (
              <div className="glass-card" style={{ padding: '24px' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '16px', color: 'var(--accent-violet)' }}>
                  📐 Mutual Information
                </h4>
                {feature_selection.mutual_information.slice(0, 10).map((f, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-secondary)', flex: 1 }}>{f.feature}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{
                        width: `${Math.max(f.mi_score * 150, 4)}px`, height: '6px',
                        background: 'linear-gradient(90deg, #a78bfa, #c084fc)', borderRadius: '3px',
                      }} />
                      <span style={{ fontWeight: 600, minWidth: '50px', textAlign: 'right' }}>{f.mi_score}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Chi-Square */}
            {feature_selection.chi_square && !feature_selection.chi_square.error && (
              <div className="glass-card" style={{ padding: '24px' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '16px', color: 'var(--accent-cyan)' }}>
                  χ² Chi-Square Test
                </h4>
                {feature_selection.chi_square.slice(0, 10).map((f, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-secondary)', flex: 1 }}>{f.feature}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 600, minWidth: '60px', textAlign: 'right' }}>{f.chi2_score}</span>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>p={f.p_value}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* SHAP Feature Table */}
      {shap_results?.feature_importance && shap_results.feature_importance.length > 0 && (
        <motion.div
          className="glass-card"
          style={{ padding: '24px', marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h4 style={{ fontWeight: 700, marginBottom: '16px' }}>📋 SHAP Feature Rankings</h4>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Feature</th>
                  <th>Mean |SHAP Value|</th>
                  <th>Relative Impact</th>
                </tr>
              </thead>
              <tbody>
                {shap_results.feature_importance.map((f, i) => {
                  const maxShap = shap_results.feature_importance[0]?.mean_shap_value || 1;
                  const pct = (f.mean_shap_value / maxShap) * 100;
                  return (
                    <tr key={i}>
                      <td style={{ fontWeight: 700, color: i < 3 ? 'var(--accent-amber)' : 'var(--text-muted)' }}>
                        {i < 3 ? ['🥇', '🥈', '🥉'][i] : `#${i + 1}`}
                      </td>
                      <td style={{ fontWeight: 600 }}>{f.feature}</td>
                      <td>{f.mean_shap_value.toFixed(4)}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{
                            width: `${Math.max(pct, 2)}%`, maxWidth: '200px', height: '8px',
                            background: 'var(--gradient-accent)', borderRadius: '4px',
                          }} />
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            {pct.toFixed(0)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
}
