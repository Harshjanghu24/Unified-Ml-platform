import { motion } from 'framer-motion';
import { HiArrowRight } from 'react-icons/hi';
import { Link } from 'react-router-dom';

const features = [
  {
    icon: '🎯',
    title: 'Binary Classification',
    desc: 'Yes/No, Pass/Fail, Disease/Healthy — handles 2-class problems with Logistic Regression, Random Forest & XGBoost.',
    color: 'var(--accent-indigo)',
    bg: 'rgba(129, 140, 248, 0.1)',
  },
  {
    icon: '🌈',
    title: 'Multi-Class Classification',
    desc: 'Iris Species, Customer Segments — supports multiple categories with Decision Trees, RF, KNN & XGBoost.',
    color: 'var(--accent-violet)',
    bg: 'rgba(167, 139, 250, 0.1)',
  },
  {
    icon: '📈',
    title: 'Regression',
    desc: 'House Prices, Salary Prediction — continuous value forecasting with Linear Regression, RF & XGBoost Regressor.',
    color: 'var(--accent-cyan)',
    bg: 'rgba(34, 211, 238, 0.1)',
  },
];

const pipeline = [
  { step: '1', title: 'Upload Dataset', desc: 'Upload any labeled CSV', icon: '📤' },
  { step: '2', title: 'AI Analysis', desc: 'Analyze all columns automatically', icon: '🔬' },
  { step: '3', title: 'Target Recommendations', desc: 'AI suggests best target columns', icon: '🧠' },
  { step: '4', title: 'Select Target', desc: 'Choose from ranked candidates', icon: '🎯' },
  { step: '5', title: 'Preprocessing', desc: 'Imputation, encoding, scaling', icon: '⚙️' },
  { step: '6', title: 'Train Models', desc: 'With hyperparameter tuning', icon: '🏋️' },
  { step: '7', title: 'Evaluate & Explain', desc: 'Metrics, SHAP, cross-validation', icon: '📊' },
  { step: '8', title: 'Predict', desc: 'Single or batch predictions', icon: '🚀' },
];

export default function HomePage() {
  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        style={{ textAlign: 'center', marginBottom: '60px', paddingTop: '20px' }}
      >
        <div style={{ marginBottom: '16px' }}>
          <span className="badge badge-binary" style={{ fontSize: '0.8rem' }}>
            Unified Supervised Learning AutoML Platform
          </span>
        </div>
        <h1 style={{ fontSize: '3rem', fontWeight: 800, lineHeight: 1.1, marginBottom: '20px' }}>
          Intelligent{' '}
          <span className="gradient-text">AutoML Platform</span>
        </h1>
        <p style={{ fontSize: '1.15rem', color: 'var(--text-secondary)', maxWidth: '700px', margin: '0 auto 32px', lineHeight: 1.7 }}>
          Upload any labeled dataset. Our AI analyzes <strong>every column</strong> and recommends
          the best target variables. Automatically detect Binary Classification, Multi-Class Classification,
          or Regression — then train, compare, and explain multiple models.
        </p>
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
          <Link to="/dataset" className="btn-primary" style={{ fontSize: '1rem', padding: '14px 32px' }}>
            Get Started <HiArrowRight />
          </Link>
          <Link to="/analyzer" className="btn-secondary" style={{ fontSize: '1rem', padding: '14px 32px' }}>
            View Analyzer
          </Link>
        </div>
      </motion.div>

      {/* Key Differentiator */}
      <motion.div
        className="glass-card"
        style={{
          padding: '32px', marginBottom: '48px', textAlign: 'center',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          background: 'rgba(99, 102, 241, 0.03)',
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.15 }}
      >
        <p style={{ fontSize: '2rem', marginBottom: '12px' }}>🧠</p>
        <h3 style={{ fontWeight: 700, fontSize: '1.2rem', marginBottom: '8px' }}>
          AI-Powered Target Recommendation
        </h3>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto', lineHeight: 1.6 }}>
          Unlike traditional AutoML tools, you <strong>don&apos;t need to specify the target column</strong>. 
          Our system analyzes entropy, variance, cardinality, and distribution metrics for every column, 
          then ranks them with confidence scores so you can make an informed selection.
        </p>
      </motion.div>

      {/* Problem Types */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 style={{
          textAlign: 'center', fontSize: '1.5rem', fontWeight: 700,
          marginBottom: '32px', color: 'var(--text-primary)'
        }}>
          Three Problem Types, One Platform
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', marginBottom: '60px' }}>
          {features.map((f, i) => (
            <motion.div
              key={i}
              className="glass-card"
              style={{ padding: '32px' }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.1 }}
              whileHover={{ y: -4 }}
            >
              <div style={{
                width: '50px', height: '50px', borderRadius: '12px',
                background: f.bg, display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: '1.5rem', marginBottom: '16px',
              }}>
                {f.icon}
              </div>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '10px', color: f.color }}>
                {f.title}
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
                {f.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* ML Pipeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <h2 style={{
          textAlign: 'center', fontSize: '1.5rem', fontWeight: 700,
          marginBottom: '32px', color: 'var(--text-primary)'
        }}>
          Complete ML Pipeline
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '16px', marginBottom: '60px' }}>
          {pipeline.map((p, i) => (
            <motion.div
              key={i}
              className="glass-card"
              style={{ padding: '20px', textAlign: 'center' }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 + i * 0.05 }}
              whileHover={{ scale: 1.03 }}
            >
              <div style={{ fontSize: '2rem', marginBottom: '8px' }}>{p.icon}</div>
              <div style={{
                fontSize: '0.7rem', fontWeight: 700, color: 'var(--accent-indigo)',
                textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px',
              }}>
                Step {p.step}
              </div>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '4px' }}>
                {p.title}
              </h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {p.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Tech Stack */}
      <motion.div
        className="glass-card"
        style={{ padding: '32px', textAlign: 'center', marginBottom: '40px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '16px' }}>
          Built With
        </h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', justifyContent: 'center' }}>
          {['React', 'Tailwind CSS', 'FastAPI', 'Scikit-Learn', 'XGBoost', 'SHAP', 'Pandas', 'Matplotlib', 'SQLite', 'Recharts'].map((tech) => (
            <span
              key={tech}
              style={{
                padding: '8px 20px',
                background: 'var(--bg-primary)',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: 500,
                color: 'var(--text-secondary)',
                border: '1px solid var(--border-color)',
              }}
            >
              {tech}
            </span>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
