import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';
import {
  HiSearchCircle, HiCheckCircle, HiArrowRight, HiExclamationCircle,
  HiStar, HiTrendingUp, HiViewGrid
} from 'react-icons/hi';
import {
  analyzeColumns, getTargetRecommendations, selectTarget
} from '../services/api.js';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from 'recharts';

const TYPE_COLORS = {
  'Binary Classification': '#818cf8',
  'Multi-Class Classification': '#a78bfa',
  'Regression': '#22d3ee',
  'Unsuitable': '#64748b',
};

const TYPE_EMOJI = {
  'Binary Classification': '🎯',
  'Multi-Class Classification': '🌈',
  'Regression': '📈',
  'Unsuitable': '⛔',
};

const RECOMMENDATION_COLORS = {
  'Excellent': '#34d399',
  'Good': '#818cf8',
  'Fair': '#fbbf24',
  'Poor': '#fb923c',
  'Unsuitable': '#64748b',
};

export default function AnalyzerPage() {
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [selectingTarget, setSelectingTarget] = useState(false);
  const [step, setStep] = useState(1); // 1=analyze, 2=recommendations, 3=selected
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const { data: analysisData } = await analyzeColumns();
      setAnalysis(analysisData);

      const { data: recData } = await getTargetRecommendations();
      setRecommendations(recData);

      setStep(2);
      toast.success(`Analysis complete! Found ${recData.best_candidates?.length || 0} excellent/good candidates.`);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Analysis failed. Make sure a dataset is uploaded.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectTarget = async (columnName) => {
    setSelectingTarget(true);
    try {
      const { data } = await selectTarget(columnName);
      setSelectedTarget(data);
      setStep(3);
      toast.success(`Target "${columnName}" selected! Problem type: ${data.problem_type.toUpperCase()}`);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Target selection failed.');
    } finally {
      setSelectingTarget(false);
    }
  };

  // Prepare chart data
  const chartData = analysis?.analysis?.filter(a => !a.is_unsuitable)?.slice(0, 10)?.map(a => ({
    name: a.column_name.length > 12 ? a.column_name.slice(0, 12) + '…' : a.column_name,
    fullName: a.column_name,
    confidence: a.confidence,
    binary: a.binary_score,
    multiclass: a.multiclass_score,
    regression: a.regression_score,
  })) || [];

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="section-header">
          <div className="icon gradient-accent" style={{ color: 'white' }}>
            <HiSearchCircle />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Dataset Analyzer</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              AI-powered analysis of every column to recommend the best target variables
            </p>
          </div>
        </div>
      </motion.div>

      {/* Step Indicator */}
      <motion.div
        className="glass-card"
        style={{ padding: '20px 32px', marginBottom: '24px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0', justifyContent: 'center' }}>
          {['Analyze Columns', 'View Recommendations', 'Select Target'].map((label, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '32px', height: '32px', borderRadius: '50%',
                  background: step > i ? 'var(--gradient-accent)' : step === i + 1 ? 'rgba(99, 102, 241, 0.2)' : 'var(--bg-primary)',
                  border: step === i + 1 ? '2px solid var(--accent-indigo)' : '2px solid transparent',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.8rem', fontWeight: 700,
                  color: step > i ? 'white' : 'var(--text-secondary)',
                  transition: 'all 0.3s ease',
                }}>
                  {step > i + 1 ? '✓' : i + 1}
                </div>
                <span style={{
                  fontSize: '0.85rem', fontWeight: step === i + 1 ? 700 : 400,
                  color: step === i + 1 ? 'var(--accent-indigo)' : 'var(--text-muted)',
                }}>
                  {label}
                </span>
              </div>
              {i < 2 && (
                <div style={{
                  width: '60px', height: '2px', margin: '0 12px',
                  background: step > i + 1 ? 'var(--accent-indigo)' : 'var(--border-color)',
                  transition: 'background 0.3s ease',
                }} />
              )}
            </div>
          ))}
        </div>
      </motion.div>

      {/* Step 1: Analyze Button */}
      {step === 1 && (
        <motion.div
          className="glass-card"
          style={{ padding: '60px', textAlign: 'center', marginBottom: '24px' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {!loading ? (
            <>
              <p style={{ fontSize: '3.5rem', marginBottom: '20px' }}>🔬</p>
              <h3 style={{ fontWeight: 700, marginBottom: '12px', fontSize: '1.3rem' }}>
                Intelligent Column Analyzer
              </h3>
              <p style={{ color: 'var(--text-muted)', marginBottom: '32px', maxWidth: '550px', margin: '0 auto 32px', lineHeight: 1.7 }}>
                Our AI will examine <strong>every column</strong> in your dataset, computing entropy,
                variance, cardinality, and distribution metrics to determine the best target candidates
                for Binary Classification, Multi-Class Classification, or Regression.
              </p>
              <button className="btn-primary" onClick={handleAnalyze} style={{ fontSize: '1.05rem', padding: '16px 44px' }}>
                <HiSearchCircle /> Start Analysis
              </button>
            </>
          ) : (
            <>
              <div className="spinner pulse-glow" style={{ margin: '0 auto 24px', width: '64px', height: '64px', borderWidth: '4px' }} />
              <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Analyzing All Columns...</h3>
              <p style={{ color: 'var(--text-muted)' }}>
                Computing entropy, variance, cardinality, and distribution scores...
              </p>
            </>
          )}
        </motion.div>
      )}

      {/* Step 2: Recommendations */}
      {step >= 2 && recommendations && (
        <>
          {/* Summary Cards */}
          <motion.div
            style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="glass-card stat-card">
              <div className="stat-value gradient-text">{recommendations.total_columns}</div>
              <div className="stat-label">Columns Analyzed</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-value" style={{ color: 'var(--accent-emerald)' }}>
                {recommendations.best_candidates?.length || 0}
              </div>
              <div className="stat-label">Best Candidates</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-value" style={{ color: 'var(--accent-amber)' }}>
                {recommendations.good_candidates?.length || 0}
              </div>
              <div className="stat-label">Fair Candidates</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-value" style={{ color: 'var(--text-muted)' }}>
                {recommendations.poor_candidates?.length || 0}
              </div>
              <div className="stat-label">Poor/Unsuitable</div>
            </div>
          </motion.div>

          {/* Confidence Chart */}
          {chartData.length > 0 && (
            <motion.div
              className="glass-card"
              style={{ padding: '24px', marginBottom: '24px' }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
            >
              <h3 style={{ fontWeight: 700, marginBottom: '20px' }}>📊 Target Suitability Scores</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chartData} barCategoryGap="15%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} angle={-20} textAnchor="end" height={60} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      color: '#f1f5f9',
                    }}
                    formatter={(value, name) => [`${value}%`, name]}
                    labelFormatter={(label) => {
                      const item = chartData.find(d => d.name === label);
                      return item?.fullName || label;
                    }}
                  />
                  <Legend />
                  <Bar dataKey="binary" name="Binary" fill="#818cf8" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="multiclass" name="Multi-Class" fill="#a78bfa" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="regression" name="Regression" fill="#22d3ee" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          )}

          {/* Main Recommendations Table */}
          <motion.div
            className="glass-card"
            style={{ padding: '24px', marginBottom: '24px' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h3 style={{ fontWeight: 700, marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <HiStar style={{ color: 'var(--accent-amber)' }} /> Target Column Recommendations
            </h3>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: '40px' }}>#</th>
                    <th>Column Name</th>
                    <th>Suggested Problem Type</th>
                    <th>Confidence</th>
                    <th>Recommendation</th>
                    <th>Unique</th>
                    <th>Missing %</th>
                    <th>Entropy</th>
                    <th style={{ width: '100px' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {recommendations.recommendations?.map((rec, i) => (
                    <tr
                      key={rec.column_name}
                      style={{
                        background: rec.recommendation === 'Excellent' ? 'rgba(52, 211, 153, 0.04)' :
                                    rec.recommendation === 'Good' ? 'rgba(129, 140, 248, 0.04)' : undefined,
                        opacity: rec.is_unsuitable ? 0.5 : 1,
                      }}
                    >
                      <td style={{ fontWeight: 600, color: 'var(--text-muted)' }}>{i + 1}</td>
                      <td style={{ fontWeight: 600 }}>{rec.column_name}</td>
                      <td>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: '6px',
                          padding: '4px 12px', borderRadius: '6px',
                          background: `${TYPE_COLORS[rec.suggested_type]}15`,
                          color: TYPE_COLORS[rec.suggested_type],
                          fontSize: '0.82rem', fontWeight: 600,
                        }}>
                          {TYPE_EMOJI[rec.suggested_type]} {rec.suggested_type}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{
                            width: '60px', height: '6px', borderRadius: '3px',
                            background: 'var(--bg-primary)',
                            overflow: 'hidden',
                          }}>
                            <div style={{
                              width: `${rec.confidence}%`, height: '100%',
                              borderRadius: '3px',
                              background: rec.confidence >= 80 ? '#34d399' :
                                          rec.confidence >= 60 ? '#818cf8' :
                                          rec.confidence >= 40 ? '#fbbf24' : '#64748b',
                              transition: 'width 0.5s ease',
                            }} />
                          </div>
                          <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{rec.confidence}%</span>
                        </div>
                      </td>
                      <td>
                        <span style={{
                          padding: '3px 10px', borderRadius: '4px',
                          background: `${RECOMMENDATION_COLORS[rec.recommendation]}20`,
                          color: RECOMMENDATION_COLORS[rec.recommendation],
                          fontSize: '0.8rem', fontWeight: 600,
                        }}>
                          {rec.recommendation}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.85rem' }}>{rec.n_unique}</td>
                      <td style={{
                        fontSize: '0.85rem',
                        color: rec.missing_pct > 10 ? 'var(--accent-rose)' : 'var(--text-secondary)',
                      }}>
                        {rec.missing_pct}%
                      </td>
                      <td style={{ fontSize: '0.85rem' }}>{rec.entropy}</td>
                      <td>
                        {!rec.is_unsuitable && rec.confidence > 20 && step === 2 && (
                          <button
                            className="btn-primary"
                            style={{
                              padding: '6px 14px', fontSize: '0.78rem',
                              opacity: selectingTarget ? 0.5 : 1,
                            }}
                            onClick={() => handleSelectTarget(rec.column_name)}
                            disabled={selectingTarget}
                          >
                            Select
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>

          {/* Detailed Column Analysis Cards */}
          {recommendations.best_candidates?.length > 0 && step === 2 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
            >
              <h3 style={{ fontWeight: 700, marginBottom: '16px' }}>🏆 Best Target Candidates</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                {recommendations.best_candidates.map((rec) => (
                  <motion.div
                    key={rec.column_name}
                    className="glass-card"
                    style={{
                      padding: '24px',
                      border: `1px solid ${TYPE_COLORS[rec.suggested_type]}40`,
                      cursor: 'pointer',
                    }}
                    whileHover={{ y: -4, boxShadow: `0 8px 30px ${TYPE_COLORS[rec.suggested_type]}20` }}
                    onClick={() => handleSelectTarget(rec.column_name)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                      <div>
                        <h4 style={{ fontWeight: 700, fontSize: '1.05rem', marginBottom: '4px' }}>
                          {rec.column_name}
                        </h4>
                        <span style={{
                          padding: '3px 10px', borderRadius: '4px',
                          background: `${TYPE_COLORS[rec.suggested_type]}15`,
                          color: TYPE_COLORS[rec.suggested_type],
                          fontSize: '0.78rem', fontWeight: 600,
                        }}>
                          {TYPE_EMOJI[rec.suggested_type]} {rec.suggested_type}
                        </span>
                      </div>
                      <div style={{
                        width: '52px', height: '52px', borderRadius: '50%',
                        background: `conic-gradient(${RECOMMENDATION_COLORS[rec.recommendation]} ${rec.confidence * 3.6}deg, var(--bg-primary) 0deg)`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        <div style={{
                          width: '40px', height: '40px', borderRadius: '50%',
                          background: 'var(--bg-secondary)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: '0.75rem', fontWeight: 700,
                        }}>
                          {rec.confidence}%
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.82rem' }}>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Unique Values:</span>
                        <span style={{ fontWeight: 600, marginLeft: '4px' }}>{rec.n_unique}</span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Missing:</span>
                        <span style={{ fontWeight: 600, marginLeft: '4px' }}>{rec.missing_pct}%</span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Entropy:</span>
                        <span style={{ fontWeight: 600, marginLeft: '4px' }}>{rec.entropy}</span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Type:</span>
                        <span style={{ fontWeight: 600, marginLeft: '4px' }}>{rec.dtype}</span>
                      </div>
                    </div>

                    {rec.top_classes && Object.keys(rec.top_classes).length > 0 && Object.keys(rec.top_classes).length <= 10 && (
                      <div style={{ marginTop: '12px' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Top Classes:</span>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '6px' }}>
                          {Object.entries(rec.top_classes).slice(0, 5).map(([cls, count]) => (
                            <span key={cls} style={{
                              padding: '2px 8px', borderRadius: '4px',
                              background: 'var(--bg-primary)',
                              fontSize: '0.75rem', fontWeight: 500,
                            }}>
                              {cls}: {count}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <div style={{ marginTop: '16px', textAlign: 'center' }}>
                      <span style={{
                        color: TYPE_COLORS[rec.suggested_type],
                        fontWeight: 600, fontSize: '0.85rem',
                      }}>
                        Click to select as target →
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </>
      )}

      {/* Step 3: Target Selected */}
      {step === 3 && selectedTarget && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="glass-card" style={{
            padding: '40px', textAlign: 'center', marginBottom: '24px',
            border: '1px solid var(--accent-emerald)',
            background: 'rgba(52, 211, 153, 0.04)',
          }}>
            <HiCheckCircle style={{ fontSize: '3.5rem', color: 'var(--accent-emerald)', marginBottom: '16px' }} />
            <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '1.3rem' }}>
              Target Column Selected!
            </h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '8px', fontSize: '1.05rem' }}>
              <strong style={{ color: 'var(--accent-indigo)' }}>{selectedTarget.target_column}</strong> →{' '}
              <span className={`badge badge-${selectedTarget.problem_type}`} style={{ fontSize: '0.85rem', padding: '4px 16px' }}>
                {selectedTarget.problem_type === 'binary' ? '🎯 Binary Classification' :
                 selectedTarget.problem_type === 'multiclass' ? '🌈 Multi-Class Classification' :
                 '📈 Regression'}
              </span>
            </p>

            {selectedTarget.target_info && (
              <div style={{
                display: 'inline-grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                gap: '16px', marginTop: '20px', maxWidth: '500px', textAlign: 'left',
              }}>
                <div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Data Type</span>
                  <p style={{ fontWeight: 600 }}>{selectedTarget.target_info.dtype}</p>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Unique Values</span>
                  <p style={{ fontWeight: 600 }}>{selectedTarget.target_info.n_unique}</p>
                </div>
                {selectedTarget.target_info.classes && (
                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Classes</span>
                    <p style={{ fontWeight: 600 }}>{selectedTarget.target_info.classes.join(', ')}</p>
                  </div>
                )}
              </div>
            )}

            <div style={{ marginTop: '32px', display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                className="btn-primary"
                onClick={() => navigate('/eda')}
                style={{ fontSize: '1rem', padding: '14px 32px' }}
              >
                View EDA <HiArrowRight />
              </button>
              <button
                className="btn-primary"
                onClick={() => navigate('/training')}
                style={{ fontSize: '1rem', padding: '14px 32px' }}
              >
                Start Training <HiArrowRight />
              </button>
              <button
                className="btn-secondary"
                onClick={() => { setStep(2); setSelectedTarget(null); }}
                style={{ fontSize: '1rem', padding: '14px 32px' }}
              >
                Choose Different Target
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
