import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Scanner from './pages/Scanner';
import Results from './pages/Results';
import HistoryPage from './pages/HistoryPage';
import BulkAnalysis from './pages/BulkAnalysis';
import NetworkGraph from './pages/NetworkGraph';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="scanner" element={<Scanner />} />
          <Route path="results" element={<Results />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="bulk" element={<BulkAnalysis />} />
          <Route path="network" element={<NetworkGraph />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
