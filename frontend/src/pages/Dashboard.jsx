import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  LayoutDashboard,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts';
import { getAnalytics } from '../services/api';

const COLORS = {
  'Low risk': '#14b8a6', // Teal
  Suspicious: '#f59e0b', // Amber
  'High risk': '#f43f5e', // Rose
  'Insufficient evidence': '#64748b', // Slate
};

export default function Dashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);
      const { data } = await getAnalytics();
      if (data) setAnalytics(data);
      setLoading(false);
    };
    fetchStats();
  }, []);

  const classData = analytics?.classification_counts
    ? Object.entries(analytics.classification_counts).map(([name, value]) => ({ name, value }))
    : [];

  const platformData = analytics?.platform_counts
    ? Object.entries(analytics.platform_counts).map(([platform, count]) => ({ platform, count }))
    : [];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <h2 className="text-xl font-semibold text-white tracking-tight flex items-center gap-2">
            <LayoutDashboard className="w-5 h-5 text-teal-400" />
            Security Threat Analytics
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Real-time aggregate detection telemetry across scanned accounts.
          </p>
        </div>

        <Link
          to="/scanner"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-teal-500 text-slate-950 font-semibold text-xs hover:bg-teal-400 transition-colors"
        >
          Scan New Account <ArrowRight className="w-4 h-4" />
        </Link>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-1">
          <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Total Scanned</span>
          <div className="text-2xl font-bold text-white font-mono">{analytics?.total_scans ?? 0}</div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-1">
          <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">High Risk Accounts</span>
          <div className="text-2xl font-bold text-rose-400 font-mono">
            {analytics?.classification_counts?.['High risk'] ?? 0}
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-1">
          <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Suspicious Accounts</span>
          <div className="text-2xl font-bold text-amber-400 font-mono">
            {analytics?.classification_counts?.['Suspicious'] ?? 0}
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-1">
          <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Average Risk Score</span>
          <div className="text-2xl font-bold text-teal-400 font-mono">
            {analytics?.average_risk_score !== null && analytics?.average_risk_score !== undefined
              ? `${analytics.average_risk_score}/100`
              : 'N/A'}
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Classification Breakdown */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 border-b border-slate-800/80 pb-3">
            Risk Tier Distribution
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {classData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[entry.name] || '#64748b'} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#e2e8f0', fontSize: '12px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Platform Breakdown */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 border-b border-slate-800/80 pb-3">
            Scans by Platform
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={platformData}>
                <XAxis dataKey="platform" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#e2e8f0', fontSize: '12px' }}
                />
                <Bar dataKey="count" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

