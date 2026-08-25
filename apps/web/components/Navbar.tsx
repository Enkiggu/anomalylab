"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, Cpu, ShieldCheck, Zap } from "lucide-react";
import { api } from "@/lib/api";

export default function Navbar() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
    refetchInterval: 10000,
  });

  const { data: currentModel } = useQuery({
    queryKey: ["currentModel"],
    queryFn: api.getCurrentModel,
  });

  return (
    <header className="h-16 border-b border-border bg-surface/80 backdrop-blur-md sticky top-0 z-40 px-6 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="h-9 w-9 rounded-lg bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center glow-cyan">
          <Activity className="h-5 w-5 text-cyan-400" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-lg tracking-tight text-white font-mono">AnomalyLab</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/30 text-cyan-400">
              PROD ML
            </span>
          </div>
          <p className="text-xs text-slate-400 font-sans hidden sm:block">Security Telemetry & Anomaly Analytics</p>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        {/* Active Champion Model Badge */}
        {currentModel && (
          <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            <span className="text-slate-400">Champion:</span>
            <span className="font-mono text-emerald-300 font-semibold">{currentModel.algorithm}</span>
            <span className="text-[10px] text-slate-500 font-mono">({currentModel.model_version.slice(-15)})</span>
          </div>
        )}

        {/* API Health Status */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="text-slate-300 font-mono">{health?.status === "READY" ? "API ONLINE" : "STANDBY"}</span>
        </div>
      </div>
    </header>
  );
}
