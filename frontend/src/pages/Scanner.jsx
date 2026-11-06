import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  AlertCircle,
  HelpCircle,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  ShieldAlert,
} from 'lucide-react';
import { scanAccount } from '../services/api';

const SAMPLES = {
  suspicious: {
    platform: 'Instagram',
    username: 'free_crypto_promo_9988',
    followers: 2,
    following: 4890,
    posts: 1,
    account_age_days: 1,
    has_profile_pic: false,
    bio: 'DM for FREE BITCOIN GIVEAWAY! Claim now: https://bit.ly/cash-fast-win',
    post_text: 'WINNER SELECTED! Tap link in bio now to claim your £1000 prize!\nWINNER SELECTED! Tap link in bio now to claim your £1000 prize!',
  },
  genuine: {
    platform: 'Instagram',
    username: 'sarah_landscape_photos',
    followers: 1850,
    following: 410,
    posts: 145,
    account_age_days: 720,
    has_profile_pic: true,
    bio: 'Pacific Northwest landscape photographer. Prints & portfolio: https://sarahphotos.org',
    post_text: 'Sunrise above the mist in Mt Rainier National Park today. Truly calm morning.',
  },
};

export default function Scanner() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    platform: 'Instagram',
    username: '',
    followers: '',
    following: '',
    posts: '',
    account_age_days: '',
    has_profile_pic: 'unknown', // 'yes' | 'no' | 'unknown'
    bio: '',
    post_text: '',
  });

  const [clientErrors, setClientErrors] = useState({});
  const [serverError, setServerError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear field-specific error
    if (clientErrors[name]) {
      setClientErrors((prev) => ({ ...prev, [name]: null }));
    }
  };

  const loadSample = (type) => {
    const s = SAMPLES[type];
    setFormData({
      platform: s.platform,
      username: s.username,
      followers: String(s.followers),
      following: String(s.following),
      posts: String(s.posts),
      account_age_days: String(s.account_age_days),
      has_profile_pic: s.has_profile_pic ? 'yes' : 'no',
      bio: s.bio,
      post_text: s.post_text,
    });
    setClientErrors({});
    setServerError(null);
  };

  const validateForm = () => {
    const errs = {};
    if (!formData.username.trim()) {
      errs.username = 'Account username or handle is required.';
    }

    const numericFields = ['followers', 'following', 'posts', 'account_age_days'];
    for (const field of numericFields) {
      if (formData[field] !== '' && formData[field] !== null) {
        const val = Number(formData[field]);
        if (isNaN(val) || val < 0) {
          errs[field] = 'Must be zero or a positive number.';
        }
      }
    }
    return errs;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setServerError(null);

    const errs = validateForm();
    if (Object.keys(errs).length > 0) {
      setClientErrors(errs);
      return;
    }

    setLoading(true);

    // Prepare payload (convert empty strings to null to avoid treating unknown as 0)
    const payload = {
      platform: formData.platform,
      username: formData.username.trim(),
      followers: formData.followers !== '' ? Number(formData.followers) : null,
      following: formData.following !== '' ? Number(formData.following) : null,
      posts: formData.posts !== '' ? Number(formData.posts) : null,
      account_age_days: formData.account_age_days !== '' ? Number(formData.account_age_days) : null,
      has_profile_pic:
        formData.has_profile_pic === 'yes' ? true : formData.has_profile_pic === 'no' ? false : null,
      bio: formData.bio.trim() ? formData.bio.trim() : null,
      post_text: formData.post_text.trim() ? formData.post_text.trim() : null,
    };

    const { data, error } = await scanAccount(payload);
    setLoading(false);

    if (error) {
      setServerError(error);
    } else if (data) {
      // Navigate to Results page with state
      navigate('/results', { state: { scanResult: data, submittedInput: payload } });
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Page Title & Samples Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <h2 className="text-xl font-semibold text-white tracking-tight flex items-center gap-2">
            <Search className="w-5 h-5 text-teal-400" />
            Social Account Authenticity Scanner
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Evaluates account metadata and text against statistical ML, NLP, and SHAP explainability.
          </p>
        </div>

        {/* Sample Helpers */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => loadSample('suspicious')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-rose-900/60 bg-rose-950/30 text-rose-300 hover:bg-rose-900/40 transition-colors"
          >
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            Load Sample Suspicious
          </button>
          <button
            type="button"
            onClick={() => loadSample('genuine')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-teal-900/60 bg-teal-950/30 text-teal-300 hover:bg-teal-900/40 transition-colors"
          >
            <ShieldCheck className="w-3.5 h-3.5 text-teal-400" />
            Load Sample Genuine
          </button>
        </div>
      </div>

      {/* Global Server Error Alert */}
      {serverError && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800 text-rose-200 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium text-rose-300">Scan Failed</p>
            <p className="text-xs text-rose-300/80 mt-0.5">{serverError}</p>
          </div>
        </div>
      )}

      {/* Main Scanner Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-6">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 border-b border-slate-800/80 pb-3">
            1. Account Identity & Platform
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Platform Selection */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Social Platform <span className="text-teal-400">*</span>
              </label>
              <select
                name="platform"
                value={formData.platform}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60"
              >
                <option value="Instagram">Instagram</option>
                <option value="Facebook">Facebook</option>
                <option value="X">X (Twitter)</option>
                <option value="YouTube">YouTube</option>
              </select>
            </div>

            {/* Username / Handle */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Username / Handle <span className="text-teal-400">*</span>
              </label>
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="e.g. handle_name"
                className={`w-full bg-slate-950 border ${
                  clientErrors.username ? 'border-rose-500' : 'border-slate-800'
                } rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60`}
              />
              {clientErrors.username && (
                <p className="text-[11px] text-rose-400 mt-1">{clientErrors.username}</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400">
              2. Quantitative Profile Metrics
            </h3>
            <span className="text-[11px] text-slate-400 italic">Optional — leave blank if unknown</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Followers */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Followers</label>
              <input
                type="number"
                name="followers"
                min="0"
                value={formData.followers}
                onChange={handleChange}
                placeholder="Unknown"
                className={`w-full bg-slate-950 border ${
                  clientErrors.followers ? 'border-rose-500' : 'border-slate-800'
                } rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60`}
              />
              {clientErrors.followers && (
                <p className="text-[11px] text-rose-400 mt-1">{clientErrors.followers}</p>
              )}
            </div>

            {/* Following */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Following</label>
              <input
                type="number"
                name="following"
                min="0"
                value={formData.following}
                onChange={handleChange}
                placeholder="Unknown"
                className={`w-full bg-slate-950 border ${
                  clientErrors.following ? 'border-rose-500' : 'border-slate-800'
                } rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60`}
              />
              {clientErrors.following && (
                <p className="text-[11px] text-rose-400 mt-1">{clientErrors.following}</p>
              )}
            </div>

            {/* Posts */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Total Posts</label>
              <input
                type="number"
                name="posts"
                min="0"
                value={formData.posts}
                onChange={handleChange}
                placeholder="Unknown"
                className={`w-full bg-slate-950 border ${
                  clientErrors.posts ? 'border-rose-500' : 'border-slate-800'
                } rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60`}
              />
              {clientErrors.posts && (
                <p className="text-[11px] text-rose-400 mt-1">{clientErrors.posts}</p>
              )}
            </div>

            {/* Account Age */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Account Age (Days)</label>
              <input
                type="number"
                name="account_age_days"
                min="0"
                value={formData.account_age_days}
                onChange={handleChange}
                placeholder="Unknown"
                className={`w-full bg-slate-950 border ${
                  clientErrors.account_age_days ? 'border-rose-500' : 'border-slate-800'
                } rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60`}
              />
              {clientErrors.account_age_days && (
                <p className="text-[11px] text-rose-400 mt-1">{clientErrors.account_age_days}</p>
              )}
            </div>
          </div>

          {/* Profile Picture Toggle */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-2">Has Profile Picture?</label>
            <div className="flex items-center gap-3">
              {[
                { label: 'Yes (Custom avatar)', value: 'yes' },
                { label: 'No (Default placeholder)', value: 'no' },
                { label: 'Unknown / Unverified', value: 'unknown' },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-xs cursor-pointer transition-colors ${
                    formData.has_profile_pic === opt.value
                      ? 'border-teal-500/60 bg-teal-500/10 text-teal-300'
                      : 'border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <input
                    type="radio"
                    name="has_profile_pic"
                    value={opt.value}
                    checked={formData.has_profile_pic === opt.value}
                    onChange={handleChange}
                    className="sr-only"
                  />
                  {opt.label}
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400">
              3. Text & Content Semantics
            </h3>
            <span className="text-[11px] text-slate-400 italic">Optional — evaluated by TF-IDF & Phishing Rules</span>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Profile Bio / Description</label>
              <textarea
                name="bio"
                rows="2"
                value={formData.bio}
                onChange={handleChange}
                placeholder="Paste account bio or about section..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60 resize-none"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Sample Post Content / Captions
              </label>
              <textarea
                name="post_text"
                rows="3"
                value={formData.post_text}
                onChange={handleChange}
                placeholder="Paste recent post captions, tweets, or messages (separate multiple posts with line breaks)..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-200 focus:outline-none focus:border-teal-500/60 resize-none"
              />
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2 px-6 py-3 rounded-lg bg-teal-500 text-slate-950 font-semibold text-sm hover:bg-teal-400 disabled:opacity-50 transition-colors shadow-sm"
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                Analysing Indicators...
              </>
            ) : (
              <>
                Run Security Analysis
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

