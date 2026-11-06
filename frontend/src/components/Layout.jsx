import React, { useState, useEffect } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  ShieldAlert,
  Search,
  LayoutDashboard,
  History,
  FileSpreadsheet,
  Share2,
  Activity,
  Menu,
  X,
  AlertTriangle,
} from 'lucide-react';
import { checkHealth } from '../services/api';

const navigationItems = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Account Scanner', path: '/scanner', icon: Search },
  { name: 'Scan History', path: '/history', icon: History },
  { name: 'Bulk Analysis', path: '/bulk', icon: FileSpreadsheet },
  { name: 'Network Graph', path: '/network', icon: Share2 },
];

export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const location = useLocation();

  useEffect(() => {
    let isMounted = true;
    const verifyBackend = async () => {
      const { data, error } = await checkHealth();
      if (isMounted) {
        if (data && data.status === 'ok') {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      }
    };

    verifyBackend();
    const interval = setInterval(verifyBackend, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen flex bg-slate-950 text-slate-100 font-sans">
      {/* Sidebar - Desktop */}
      <aside className="hidden md:flex md:w-64 flex-col border-r border-slate-800/80 bg-slate-900/60 backdrop-blur-md">
        <div className="h-16 flex items-center gap-3 px-6 border-b border-slate-800/80">
          <ShieldAlert className="w-6 h-6 text-teal-400" />
          <div className="flex flex-col">
            <span className="font-semibold tracking-tight text-white leading-tight">SocialShield AI</span>
            <span className="text-[10px] text-slate-400 font-mono tracking-wider uppercase">Inauthentic Defense</span>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-teal-500/10 text-teal-300 border border-teal-500/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                {item.name}
              </NavLink>
            );
          })}
        </nav>

        {/* Ethical System Notice */}
        <div className="p-4 m-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
          <p className="font-medium text-slate-300 mb-1 flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-teal-400" />
            Ethical Heuristics Active
          </p>
          All classifications are probabilistic suspicious indicators. Not proof of user identity.
        </div>
      </aside>

      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="h-16 border-b border-slate-800/80 bg-slate-900/40 backdrop-blur-md flex items-center justify-between px-4 sm:px-8">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="md:hidden p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            >
              {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <h1 className="text-sm font-medium text-slate-300 hidden sm:block">
              {navigationItems.find((i) => i.path === location.pathname)?.name || 'Security Center'}
            </h1>
          </div>

          <div className="flex items-center gap-4">
            {/* Backend health status badge */}
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-mono border border-slate-800 bg-slate-950">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus === 'online'
                    ? 'bg-teal-400 animate-pulse'
                    : backendStatus === 'checking'
                    ? 'bg-amber-400 animate-ping'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-300">
                {backendStatus === 'online' ? 'API Online' : backendStatus === 'checking' ? 'Connecting...' : 'API Offline'}
              </span>
            </div>
          </div>
        </header>

        {/* Backend Offline Warning Banner */}
        {backendStatus === 'offline' && (
          <div className="bg-rose-950/80 border-b border-rose-800 text-rose-200 px-4 py-2.5 text-xs flex items-center justify-center gap-2 font-medium">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            Backend API unreachable on port 8000. Start backend using: <code className="bg-rose-900/50 px-1 py-0.5 rounded font-mono">uvicorn app.main:app --reload</code>
          </div>
        )}

        {/* Mobile Navigation Drawer */}
        {isSidebarOpen && (
          <div className="md:hidden border-b border-slate-800 bg-slate-900 px-4 py-3 space-y-1">
            {navigationItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setIsSidebarOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium ${
                    isActive ? 'bg-teal-500/10 text-teal-300' : 'text-slate-400 hover:text-slate-200'
                  }`
                }
              >
                <item.icon className="w-4 h-4" />
                {item.name}
              </NavLink>
            ))}
          </div>
        )}

        {/* Page Content */}
        <main className="flex-1 p-4 sm:p-8 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

