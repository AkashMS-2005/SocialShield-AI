import React from 'react';
import { Share2, Construction, Network } from 'lucide-react';

export default function NetworkGraph() {
  return (
    <div className="max-w-4xl mx-auto py-12 text-center space-y-6">
      <div className="w-16 h-16 rounded-2xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center mx-auto">
        <Network className="w-8 h-8" />
      </div>

      <div className="space-y-2">
        <h2 className="text-xl font-bold text-white tracking-tight">Coordinated Inauthentic Network Graph</h2>
        <p className="text-sm text-slate-400 max-w-lg mx-auto">
          Graph clustering and follower/following interaction analysis (NetworkX engine). Scheduled for upcoming module implementation.
        </p>
      </div>

      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-400 font-mono">
        <Construction className="w-3.5 h-3.5 text-amber-400" />
        Module: Coordinated Cluster Detection (Coming in Next Step)
      </div>
    </div>
  );
}

