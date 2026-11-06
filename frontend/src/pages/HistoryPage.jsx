import React, { useState, useEffect } from 'react';
import { History, Filter, Search, ArrowRight } from 'lucide-react';
import { getHistory } from '../services/api';

export default function HistoryPage() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [platform, setPlatform] = useState('');
  const [classification, setClassification] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchItems = async () => {
    setLoading(true);
    const { data } = await getHistory(page, 15, platform || null, classification || null);
    if (data) {
      setItems(data.items || []);
      setTotal(data.total || 0);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchItems();
  }, [page, platform, classification]);

  const getBadgeStyle = (cls) => {
    switch (cls) {
      case 'High risk':
        return 'bg-rose-500/10 border-rose-500/30 text-rose-300';
      case 'Suspicious':
        return 'bg-amber-500/10 border-amber-500/30 text-amber-300';
      case 'Low risk':
        return 'bg-teal-500/10 border-teal-500/30 text-teal-300';
      default:
        return 'bg-slate-800 border-slate-700 text-slate-400';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-white tracking-tight flex items-center gap-2">
            <History className="w-5 h-5 text-teal-400" />
            Scan Activity Log
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Persisted history of completed scans and classification records.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-3">
          <select
            value={platform}
            onChange={(e) => {
              setPlatform(e.target.value);
              setPage(1);
            }}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-teal-500"
          >
            <option value="">All Platforms</option>
            <option value="Instagram">Instagram</option>
            <option value="Facebook">Facebook</option>
            <option value="X">X (Twitter)</option>
            <option value="YouTube">YouTube</option>
          </select>

          <select
            value={classification}
            onChange={(e) => {
              setClassification(e.target.value);
              setPage(1);
            }}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-teal-500"
          >
            <option value="">All Risk Tiers</option>
            <option value="Low risk">Low risk</option>
            <option value="Suspicious">Suspicious</option>
            <option value="High risk">High risk</option>
            <option value="Insufficient evidence">Insufficient evidence</option>
          </select>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 font-mono uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Platform</th>
                <th className="py-3 px-4">Username</th>
                <th className="py-3 px-4">Classification</th>
                <th className="py-3 px-4 text-right">Risk Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {items.length > 0 ? (
                items.map((row) => (
                  <tr key={row.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 px-4 font-mono text-slate-400">
                      {new Date(row.scanned_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 font-medium text-slate-200">{row.platform}</td>
                    <td className="py-3 px-4 font-mono text-teal-300">{row.username}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${getBadgeStyle(row.classification)}`}>
                        {row.classification}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono font-semibold">
                      {row.risk_score !== null ? `${row.risk_score}/100` : '—'}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="py-8 text-center text-slate-500 italic">
                    {loading ? 'Loading scan history...' : 'No scan history recorded matching filters.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination footer */}
        <div className="p-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span>Total records: {total}</span>
          <div className="flex gap-2">
            <button
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(p - 1, 1))}
              className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40"
            >
              Previous
            </button>
            <span className="py-1 px-2 font-mono">Page {page}</span>
            <button
              disabled={page * 15 >= total}
              onClick={() => setPage((p) => p + 1)}
              className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

