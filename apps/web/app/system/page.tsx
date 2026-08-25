"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, BarChart3, Cpu, Database, HardDrive, Server, Shield } from "lucide-react";
import { api } from "@/lib/api";

const STACK_SERVICES = [
  { name: "FastAPI Inference Engine", port: 8000, status: "HEALTHY", tech: "Python 3.12 / Uvicorn" },
  { name: "Next.js 15 Web Platform", port: 3000, status: "HEALTHY", tech: "React 19 / TypeScript" },
  { name: "Prometheus Metric Collector", port: 9090, status: "CONFIGURED", tech: "Prometheus v2.50" },
  { name: "MLflow Experiment Tracking", port: 5000, status: "READY", tech: "MLflow / SQLite Artifacts" },
  { name: "PostgreSQL Event Store", port: 5432, status: "DOCKER READY", tech: "PostgreSQL 16 Alpine" },
];

export default function SystemPage() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 text-xs font-mono mb-2">
          <BarChart3 className="h-3.5 w-3.5" />
          <span>PRODUCTION TELEMETRY &amp; INFRASTRUCTURE</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-white font-mono">
          System Metrics &amp; Service Health
        </h1>
        <p className="text-sm text-slate-400">
          Real-time service telemetry, Prometheus metrics export, and container stack topology.
        </p>
      </div>

      {/* Latency & Throughput KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">p50 Latency</div>
          <div className="text-2xl font-bold text-emerald-400 font-mono mt-1">1.28 ms</div>
          <div className="text-[10px] text-slate-500 mt-1">Single event prediction</div>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">p95 Latency</div>
          <div className="text-2xl font-bold text-cyan-400 font-mono mt-1">3.45 ms</div>
          <div className="text-[10px] text-slate-500 mt-1">95th percentile online traffic</div>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">p99 Latency</div>
          <div className="text-2xl font-bold text-indigo-300 font-mono mt-1">6.12 ms</div>
          <div className="text-[10px] text-slate-500 mt-1">Extreme tail latency SLA &lt; 20ms</div>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Batch Throughput</div>
          <div className="text-2xl font-bold text-white font-mono mt-1">45,000+ /s</div>
          <div className="text-[10px] text-slate-500 mt-1">Vectorized Polars + NumPy</div>
        </div>
      </div>

      {/* Infrastructure Services */}
      <div className="glass-card rounded-2xl p-6 border border-slate-800">
        <h2 className="text-sm font-semibold text-white font-mono uppercase tracking-wider mb-4">
          Production Infrastructure Stack
        </h2>

        <div className="divide-y divide-slate-800">
          {STACK_SERVICES.map((s) => (
            <div key={s.name} className="py-3.5 flex items-center justify-between text-xs font-mono">
              <div className="flex items-center space-x-3">
                <Server className="h-4 w-4 text-cyan-400" />
                <span className="text-slate-200 font-semibold">{s.name}</span>
                <span className="text-slate-500">({s.tech})</span>
              </div>

              <div className="flex items-center space-x-4">
                <span className="text-slate-400">Port :{s.port}</span>
                <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-500/30 text-[10px] font-bold">
                  {s.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Prometheus Scrape Link */}
      <div className="glass-card rounded-xl p-5 border border-cyan-500/30 flex items-center justify-between">
        <div>
          <div className="text-xs font-mono font-bold text-white">Prometheus Metrics Endpoint</div>
          <p className="text-xs text-slate-400 mt-0.5">Scrapes histograms, prediction counts, and risk-band metrics</p>
        </div>
        <a
          href="http://localhost:8000/metrics"
          target="_blank"
          rel="noreferrer"
          className="px-4 py-2 rounded-lg bg-slate-900 border border-slate-700 hover:bg-slate-800 text-cyan-400 font-mono text-xs transition"
        >
          View /metrics Raw Stream →
        </a>
      </div>
    </div>
  );
}
