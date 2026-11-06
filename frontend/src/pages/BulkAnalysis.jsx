import React, { useState } from 'react';
import { FileSpreadsheet, Upload, AlertCircle, CheckCircle2 } from 'lucide-react';
import { bulkScanCsv } from '../services/api';

export default function BulkAnalysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a CSV file to upload.');
      return;
    }

    setLoading(true);
    setError(null);
    const { data, error: apiErr } = await bulkScanCsv(file);
    setLoading(false);

    if (apiErr) {
      setError(apiErr);
    } else if (data) {
      setResults(data);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="border-b border-slate-800/80 pb-4">
        <h2 className="text-xl font-semibold text-white tracking-tight flex items-center gap-2">
          <FileSpreadsheet className="w-5 h-5 text-teal-400" />
          Bulk Batch Analysis (CSV Upload)
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload multi-platform CSV exports. Platform headers (followers, subscribers, following) are mapped automatically.
        </p>
      </div>

      {/* Upload Zone */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center space-y-4">
        <Upload className="w-10 h-10 text-teal-400 mx-auto" />
        <div className="max-w-md mx-auto space-y-1">
          <p className="text-sm font-medium text-slate-200">Select CSV file for batch scanning</p>
          <p className="text-xs text-slate-400">
            Accepts standard headers: <code className="text-teal-300 font-mono">platform, username, followers, following, posts, bio</code>
          </p>
        </div>

        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-teal-500/10 file:text-teal-300 hover:file:bg-teal-500/20 cursor-pointer"
        />

        {error && (
          <div className="p-3 bg-rose-950/40 border border-rose-800 text-rose-300 text-xs rounded-lg max-w-md mx-auto">
            {error}
          </div>
        )}

        <div>
          <button
            onClick={handleUpload}
            disabled={loading || !file}
            className="px-6 py-2.5 rounded-lg bg-teal-500 text-slate-950 font-semibold text-xs hover:bg-teal-400 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Processing Accounts...' : 'Process Batch File'}
          </button>
        </div>
      </div>

      {/* Results Table if uploaded */}
      {results && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white">Processed Batch Results ({results.total} accounts)</h3>
            <span className="text-xs font-mono text-teal-400">All records persisted to database</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 font-mono uppercase tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Username</th>
                  <th className="py-2.5 px-3">Platform</th>
                  <th className="py-2.5 px-3">Classification</th>
                  <th className="py-2.5 px-3 text-right">Risk Score</th>
                  <th className="py-2.5 px-3">Top Recommendation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {results.results.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-3 font-mono text-teal-300">{row.username}</td>
                    <td className="py-2.5 px-3">{row.platform}</td>
                    <td className="py-2.5 px-3 font-medium">{row.classification}</td>
                    <td className="py-2.5 px-3 text-right font-mono font-semibold">
                      {row.risk_score !== null ? `${row.risk_score}/100` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 truncate max-w-xs">
                      {row.recommendations?.[0] || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

