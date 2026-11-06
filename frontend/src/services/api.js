/**
 * services/api.js – Centralised HTTP service with Axios.
 * Manages baseURL, error parsing, and API contracts.
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Centralised error extractor
const formatError = (error) => {
  if (error.response) {
    // Backend returned 4xx or 5xx
    const data = error.response.data;
    if (data?.details && Array.isArray(data.details)) {
      return data.details.map((d) => `${d.field}: ${d.message}`).join(', ');
    }
    return data?.message || data?.detail || `Server error (${error.response.status})`;
  } else if (error.request) {
    // Request made but no response (backend down)
    return 'Unable to connect to the backend server. Please verify the FastAPI service is running on http://localhost:8000.';
  } else {
    return error.message || 'An unexpected error occurred.';
  }
};

/**
 * Health check
 */
export const checkHealth = async () => {
  try {
    const res = await client.get('/api/health');
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: formatError(err) };
  }
};

/**
 * Single account scan
 */
export const scanAccount = async (accountData) => {
  try {
    const res = await client.post('/api/scan', accountData);
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: formatError(err) };
  }
};

/**
 * Standalone NLP text analysis
 */
export const analyzeText = async (text, platform = null) => {
  try {
    const res = await client.post('/api/analyze-text', { text, platform });
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: formatError(err) };
  }
};

/**
 * Bulk scan JSON list
 */
export const bulkScan = async (accounts) => {
  try {
    const res = await client.post('/api/bulk-scan', { accounts });
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: formatError(err) };
  }
};

/**
 * Bulk scan CSV upload
 */
export const bulkScanCsv = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await client.post('/api/bulk-scan-csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: formatError(err) };
  }
};

/**
 * Scan history
 */
export const getHistory = async (page = 1, limit = 20, platform = null, classification = null) => {
  try {
    const params = { page, limit };
    if (platform) params.platform = platform;
    if (classification) params.classification = classification;
    const res = await client.get('/api/history', { params });
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: formatError(err) };
  }
};

/**
 * Analytics
 */
export const getAnalytics = async () => {
  try {
    const res = await client.get('/api/analytics');
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: formatError(err) };
  }
};

export default client;

