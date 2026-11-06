import React from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  HelpCircle,
  TrendingUp,
  FileText,
  Activity,
  Layers,
  Info,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export default function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const scanResult = location.state?.scanResult;

  if (!scanResult) {
    return (
      <div className="max-w-xl mx-auto text-center py-16 space-y-4">
        <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto" />
        <h2 className="text-lg font-semibold text-white">No Scan Result Found</h2>
        <p className="text-sm text-slate-400">
          Please run a scan from the Scanner page or select a record from Scan History.
        </p>
        <Link
          to="/scanner"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-teal-500 text-slate-950 text-sm font-semibold hover:bg-teal-400"
        >
          <ArrowLeft className="w-4 h-4" /> Go to Scanner
        </Link>
      </div>
    );
  }

  const {
    username,
    platform,
    classification,
    risk_score,
    ml_result,
    nlp_result,
    suspicious_indicators,
    explainability,
    recommendations,
    limitations,
    data_availability,
    scanned_at,
  } = scanResult;

  // Format SHAP data for bar chart
  const chartData = (explainability?.top_features || []).map((f) => ({
    name: f.display_name || f.feature,
    impact: Math.round(f.shap_value * 100) / 100,
    direction: f.direction,
    sentence: f.sentence,
  }));

  // Style badge based on ethical classification
  const getBadgeStyle = (cls) => {
    switch (cls) {
      case 'High risk':
        return 'bg-rose-500/10 border-rose-500/30 text-rose-300';
      case 'Suspicious':
        return 'bg-amber-500/10 border-amber-500/30 text-amber-300';
      case 'Low risk':
        return 'bg-teal-500/10 border-teal-500/30 text-teal-300';
      default:
        return 'bg-slate-800 border-slate-700 text-slate-300';
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Back Button & Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <button
          onClick={() => navigate('/scanner')}
          className="flex items-center gap-2 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Scanner
        </button>
        <span className="text-xs font-mono text-slate-400">
          Scanned at: {new Date(scanned_at).toLocaleString()}
        </span>
      </div>

      {/* Top Hero: Account & Composite Risk Score */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Account Identity */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
          <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">Target Profile</span>
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">{username}</h2>
            <p className="text-xs font-mono text-teal-400 mt-0.5">{platform} Network</p>
          </div>

          <div className="pt-2 border-t border-slate-800/80 space-y-2 text-xs text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Data Availability:</span>
              <span className="font-mono">{Math.round(data_availability * 100)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">ML Model:</span>
              <span className="font-mono text-slate-200">{ml_result?.model_used || 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* Middle: Classification & Score */}
        <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <div>
              <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">Composite Risk Level</span>
              <div className="flex items-center gap-3 mt-1.5">
                <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getBadgeStyle(classification)}`}>
                  {classification}
                </span>
              </div>
            </div>

            <div className="text-right">
              <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">Risk Score</span>
              <div className="text-3xl font-extrabold text-white mt-0.5 font-mono">
                {risk_score !== null ? `${risk_score}/100` : 'N/A'}
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800/80 text-xs text-slate-300 flex items-center gap-2">
            <Info className="w-4 h-4 text-slate-400 shrink-0" />
            <span>
              Outputs are probabilistic indicators of inauthentic patterns, not proof of individual identity.
            </span>
          </div>
        </div>
      </div>

      {/* SHAP Feature Contribution Chart */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-teal-400" />
            <h3 className="text-sm font-semibold text-white">SHAP Feature Influence Attribution</h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">Local Explanation</span>
        </div>

        {chartData.length > 0 ? (
          <>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                  <XAxis type="number" stroke="#64748b" fontSize={11} />
                  <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={11} width={140} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ color: '#e2e8f0', fontSize: '12px' }}
                    formatter={(value) => [`${value} impact`, 'SHAP Score']}
                  />
                  <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.direction === 'toward suspicious' ? '#f43f5e' : '#14b8a6'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Explanation Sentences List */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3 border-t border-slate-800/80">
              {explainability.top_features.map((f, i) => (
                <div key={i} className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300">
                  <span className="font-semibold text-slate-200 block mb-1">{f.display_name}:</span>
                  {f.sentence}
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-xs text-slate-400 italic">No feature attribution available for sparse profiles.</p>
        )}
      </div>

      {/* Multimodal Decomposition: ML, NLP, Heuristics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* ML & NLP Results */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800/80 pb-3">
            <Layers className="w-4 h-4 text-teal-400" />
            <h3 className="text-sm font-semibold text-white">Sub-Model Diagnostics</h3>
          </div>

          <div className="space-y-4 text-xs">
            {/* ML Box */}
            <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="font-medium text-slate-300">Tabular ML Probability:</span>
                <span className="font-mono text-teal-400">
                  {ml_result?.probability !== null ? `${Math.round(ml_result.probability * 100)}%` : 'Unavailable'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Calibrated probability based on activity frequency and follower ratios.
              </p>
            </div>

            {/* NLP Box */}
            <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="font-medium text-slate-300">NLP Text Spam Score:</span>
                <span className="font-mono text-teal-400">
                  {nlp_result?.spam_score !== null ? `${Math.round(nlp_result.spam_score * 100)}%` : 'Unavailable'}
                </span>
              </div>
              {nlp_result?.suspicious_phrases?.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {nlp_result.suspicious_phrases.map((phrase, idx) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-rose-950/60 border border-rose-800/80 text-[10px] text-rose-300 font-mono">
                      {phrase}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Suspicious Indicators Triggered */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800/80 pb-3">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-white">Suspicious Indicators</h3>
          </div>

          {suspicious_indicators?.length > 0 ? (
            <ul className="space-y-2 text-xs text-slate-300">
              {suspicious_indicators.map((ind, idx) => (
                <li key={idx} className="flex items-start gap-2 p-2 rounded bg-slate-950/40 border border-slate-800/60">
                  <span className="text-amber-400 shrink-0 mt-0.5">•</span>
                  <span>{ind}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-slate-400 italic">No specific rule violations or spam flags triggered.</p>
          )}
        </div>
      </div>

      {/* Recommendations & Limitations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Recommendations */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-3">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 border-b border-slate-800/80 pb-2">
            Actionable Recommendations
          </h3>
          <ul className="space-y-2 text-xs text-slate-300">
            {recommendations?.map((rec, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-teal-400 shrink-0 mt-0.5" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Limitations (Ethical Rule #4) */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-3">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 border-b border-slate-800/80 pb-2">
            Assessment Limitations
          </h3>
          <ul className="space-y-1.5 text-xs text-slate-400">
            {limitations?.map((lim, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-slate-500 font-mono">—</span>
                <span>{lim}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

