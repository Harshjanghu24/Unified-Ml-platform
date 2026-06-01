import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Sidebar from './components/Sidebar.jsx';
import HomePage from './pages/HomePage.jsx';
import DatasetPage from './pages/DatasetPage.jsx';
import AnalyzerPage from './pages/AnalyzerPage.jsx';
import EDAPage from './pages/EDAPage.jsx';
import TrainingPage from './pages/TrainingPage.jsx';
import EvaluationPage from './pages/EvaluationPage.jsx';
import ExplainabilityPage from './pages/ExplainabilityPage.jsx';
import PredictionPage from './pages/PredictionPage.jsx';
import SavedModelsPage from './pages/SavedModelsPage.jsx';

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: theme === 'dark' ? '#1e293b' : '#ffffff',
            color: theme === 'dark' ? '#f1f5f9' : '#0f172a',
            border: '1px solid var(--border-color)',
            borderRadius: '10px',
            fontSize: '0.9rem',
          },
        }}
      />

      <div style={{ display: 'flex', minHeight: '100vh' }}>
        <Sidebar
          collapsed={collapsed}
          setCollapsed={setCollapsed}
          theme={theme}
          setTheme={setTheme}
        />

        <main
          style={{
            flex: 1,
            marginLeft: collapsed ? '72px' : '260px',
            transition: 'margin-left 0.3s ease',
            padding: '32px',
            minHeight: '100vh',
          }}
        >
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/analyzer" element={<AnalyzerPage />} />
            <Route path="/eda" element={<EDAPage />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/explainability" element={<ExplainabilityPage />} />
            <Route path="/prediction" element={<PredictionPage />} />
            <Route path="/models" element={<SavedModelsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
