"use client";

import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Cpu,
  Layers,
  Percent,
  Radio,
  Shield,
  ShieldCheck,
  Zap,
} from "lucide-react";
import Link from "next/link";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";

const mockTrendData = [
  { time: "00:00", normal: 120, anomaly: 2 },
  { time: "04:00", normal: 90, anomaly: 4 },
  { time: "08:00", normal: 340, anomaly: 3 },
  { time: "12:00", normal: 520, anomaly: 12 },
  { time: "16:00", normal: 480, anomaly: 8 },
  { time: "20:00", normal: 260, anomaly: 5 },
  { time: "23:59", normal: 150, anomaly: 3 },
];

export default function DashboardPage() {
  const { data: currentModel } = useQuery({
    queryKey: ["currentModel"],
    queryFn: api.getCurrentModel,
  });

  const { data: models = [] } = useQuery({
    queryKey: ["models"],
    queryFn: api.getModels,
  });

  const { data: drift } = useQuery({
    queryKey: ["drift"],
    queryFn: api.getDriftReport,
  });

  const metrics = currentModel?.metrics || {};
  const prAuc = metrics.pr_auc ?? 0.8015;
  const recall = metrics.recall ?? 0.8827;
  const precision = metrics.precision ?? 0.7783;
  const fpr = metrics.false_positive_rate ?? 0.0055;
  const fpPer10k = metrics.fp_per_10k ?? 55.4;

  return (
    <div className="space-y-8">
      {/* Top Banner: Champion Model Info */}
      <div className="glass-card rounded-2xl p-6 md:p-8 relative overflow-hidden border border-cyan-500/30">
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div>
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 text-xs font-mono mb-3">
              <ShieldCheck className="h-3.5 w-3.5" />
              <span>ACTIVE PRODUCTION CHAMPION</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-white font-mono tracking-tight">
              {currentModel ? currentModel.algorithm : "Isolation Forest"} Anomaly Detector
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Model Version: <span className="font-mono text-cyan-300">{currentModel?.model_version || "iforest_v1"}</span> • Feature Set: <span className="font-mono text-slate-300">v1 (15 features)</span>
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <Link
              href="/predictions"
              className="px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-semibold text-xs flex items-center space-x-2 transition shadow-lg shadow-cyan-500/20"
            >
              <Zap className="h-4 w-4" />
              <span>Test Live Inference</span>
            </Link>
            <Link
              href="/experiments"
              className="px-4 py-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs flex items-center space-x-1.5 transition"
            >
              <span>View Evaluation</span>
              <ArrowUpRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>PR-AUC (Precision-Recall)</span>
            <Shield className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold text-cyan-400 font-mono mt-2">
            {(prAuc * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Primary metric for class imbalance</p>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Recall @ Threshold</span>
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold text-emerald-400 font-mono mt-2">
            {(recall * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Target &gt;= 90% (Threshold: {currentModel?.threshold.toFixed(4) || "0.7227"})
          </p>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Precision</span>
            <Percent className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold text-indigo-300 font-mono mt-2">
            {(precision * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">F1: {(metrics.f1 ?? 0.8272).toFixed(4)}</p>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>False Positive Burden</span>
            <AlertTriangle className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold text-amber-400 font-mono mt-2">
            {fpPer10k.toFixed(1)}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">False alarms per 10,000 normal events</p>
        </div>
      </div>

      {/* Chart Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Anomaly Trend Area Chart */}
        <div className="glass-card rounded-xl p-6 border border-slate-800 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-white font-mono uppercase tracking-wider">
                Telemetry Anomaly Distribution (24h Stream)
              </h2>
              <p className="text-xs text-slate-400">Normal vs Flagged Anomalous Ingestion Volumes</p>
            </div>
            <div className="flex items-center space-x-3 text-xs font-mono">
              <span className="flex items-center space-x-1 text-slate-400">
                <span className="h-2.5 w-2.5 rounded-full bg-cyan-500 inline-block" />
                <span>Normal</span>
              </span>
              <span className="flex items-center space-x-1 text-rose-400">
                <span className="h-2.5 w-2.5 rounded-full bg-rose-500 inline-block" />
                <span>Anomaly</span>
              </span>
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockTrendData}>
                <defs>
                  <linearGradient id="colorNormal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorAnomaly" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={11} fontStyle="monospace" />
                <YAxis stroke="#64748b" fontSize={11} fontStyle="monospace" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: "12px" }}
                />
                <Area type="monotone" dataKey="normal" stroke="#06b6d4" fillOpacity={1} fill="url(#colorNormal)" />
                <Area type="monotone" dataKey="anomaly" stroke="#ef4444" fillOpacity={1} fill="url(#colorAnomaly)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Fleet & Drift Status */}
        <div className="space-y-6">
          {/* Drift Status Card */}
          <div className="glass-card rounded-xl p-5 border border-slate-800">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Drift Monitor</span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                  drift?.overall_status === "NO_DRIFT"
                    ? "bg-emerald-950/80 border-emerald-500/30 text-emerald-400"
                    : "bg-amber-950/80 border-amber-500/30 text-amber-400"
                }`}
              >
                {drift?.overall_status || "NO_DRIFT"}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Tested against reference train set (KS-test &amp; PSI 10-bin distribution).
            </p>
            <div className="mt-3 text-xs text-slate-500 font-mono">
              Features checked: <span className="text-slate-200">{drift?.features_evaluated ?? 15}</span> • Drifting: <span className="text-emerald-400">{drift?.drifting_features_count ?? 0}</span>
            </div>
            <Link
              href="/drift"
              className="mt-4 block text-center text-xs font-mono text-cyan-400 hover:text-cyan-300 py-1.5 bg-slate-900/60 rounded-lg border border-slate-800 transition"
            >
              Inspect Drift Distributions →
            </Link>
          </div>

          {/* Model Fleet Summary */}
          <div className="glass-card rounded-xl p-5 border border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Model Registry Fleet</span>
              <Link href="/models" className="text-xs text-cyan-400 hover:underline">
                View All ({models.length})
              </Link>
            </div>

            <div className="space-y-2.5">
              {models.slice(0, 3).map((m) => (
                <div
                  key={m.model_version}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 text-xs"
                >
                  <div>
                    <div className="font-mono text-slate-200 font-medium">{m.algorithm}</div>
                    <div className="text-[10px] text-slate-500 font-mono">PR-AUC: {((m.metrics?.pr_auc || 0) * 100).toFixed(1)}%</div>
                  </div>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded uppercase ${
                      m.stage === "production"
                        ? "bg-emerald-950 text-emerald-400 border border-emerald-500/30"
                        : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {m.stage}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
