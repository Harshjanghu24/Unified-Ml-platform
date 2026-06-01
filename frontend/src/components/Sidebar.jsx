import { NavLink } from 'react-router-dom';
import {
  HiHome, HiDatabase, HiChartBar, HiDocumentReport,
  HiCursorClick, HiSave, HiSun, HiMoon, HiChevronLeft, HiChevronRight,
  HiBeaker, HiSearchCircle, HiLightBulb
} from 'react-icons/hi';

const navItems = [
  { path: '/', icon: HiHome, label: 'Home' },
  { path: '/dataset', icon: HiDatabase, label: 'Dataset Upload' },
  { path: '/analyzer', icon: HiSearchCircle, label: 'Dataset Analyzer' },
  { path: '/eda', icon: HiChartBar, label: 'EDA Dashboard' },
  { path: '/training', icon: HiBeaker, label: 'Training' },
  { path: '/evaluation', icon: HiDocumentReport, label: 'Evaluation' },
  { path: '/explainability', icon: HiLightBulb, label: 'Explainability' },
  { path: '/prediction', icon: HiCursorClick, label: 'Prediction' },
  { path: '/models', icon: HiSave, label: 'Saved Models' },
];

export default function Sidebar({ collapsed, setCollapsed, theme, setTheme }) {
  return (
    <aside
      style={{
        width: collapsed ? '72px' : '260px',
        minHeight: '100vh',
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border-color)',
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.3s ease',
        position: 'fixed',
        top: 0,
        left: 0,
        zIndex: 50,
        overflow: 'hidden',
      }}
    >
      {/* Logo */}
      <div
        style={{
          padding: collapsed ? '20px 16px' : '20px 24px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          minHeight: '72px',
        }}
      >
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            background: 'var(--gradient-accent)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <HiBeaker style={{ color: 'white', fontSize: '18px' }} />
        </div>
        {!collapsed && (
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.95rem', lineHeight: 1.2 }}>
              AutoML
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Platform
            </div>
          </div>
        )}
      </div>

      {/* Nav Items */}
      <nav style={{ flex: 1, padding: '12px 8px', display: 'flex', flexDirection: 'column', gap: '4px', overflowY: 'auto' }}>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: collapsed ? '12px 16px' : '10px 16px',
              borderRadius: '8px',
              textDecoration: 'none',
              color: isActive ? 'var(--accent-indigo)' : 'var(--text-secondary)',
              background: isActive ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
              fontWeight: isActive ? 600 : 400,
              fontSize: '0.87rem',
              transition: 'all 0.2s ease',
              justifyContent: collapsed ? 'center' : 'flex-start',
              whiteSpace: 'nowrap',
            })}
          >
            <item.icon style={{ fontSize: '20px', flexShrink: 0 }} />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Bottom Actions */}
      <div style={{ padding: '12px 8px', borderTop: '1px solid var(--border-color)' }}>
        {/* Theme Toggle */}
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '10px 16px',
            borderRadius: '8px',
            border: 'none',
            background: 'transparent',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            width: '100%',
            fontSize: '0.9rem',
            justifyContent: collapsed ? 'center' : 'flex-start',
            transition: 'all 0.2s ease',
          }}
        >
          {theme === 'dark' ? <HiSun style={{ fontSize: '20px' }} /> : <HiMoon style={{ fontSize: '20px' }} />}
          {!collapsed && <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>}
        </button>

        {/* Collapse Toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '10px 16px',
            borderRadius: '8px',
            border: 'none',
            background: 'transparent',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            width: '100%',
            fontSize: '0.9rem',
            justifyContent: collapsed ? 'center' : 'flex-start',
            transition: 'all 0.2s ease',
          }}
        >
          {collapsed ? <HiChevronRight style={{ fontSize: '20px' }} /> : <HiChevronLeft style={{ fontSize: '20px' }} />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
