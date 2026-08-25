"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle,
  Clock,
  GitBranch,
  Play,
  RefreshCw,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { api } from "@/lib/api";

export default function DriftPage() {
  const queryClient = useQueryClient();

  const { data: drift, isLoading } = useQuery({
    queryKey: ["drift"],
    queryFn: api.getDriftReport,
  });

  const runDriftMutation = useMutation({
    mutationFn: api.runDriftCheck,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift"] });
    },
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "DRIFT_DETECTED":
        return "bg-rose-950 text-rose-400 border-rose-500/40";
      case "WARNING":
        return "bg-amber-950 text-amber-400 border-amber-500/40";
      default:
        return "bg-emerald-950 text-emerald-400 border-emerald-500/40";
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 text-xs font-mono mb-2">
            <GitBranch className="h-3.5 w-3.5" />
            <span>STATISTICAL DATA &amp; SCORE DRIFT MONITOR</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-white font-mono">
            Feature &amp; Concept Drift Analytics
          </h1>
          <p className="text-sm text-slate-400">
            Kolmogorov-Smirnov 2-sample tests, Population Stability Index (PSI), and Jensen-Shannon divergence.
          </p>
        </div>

        <button
          disabled={runDriftMutation.isPending}
          onClick={() => runDriftMutation.mutate()}
          className="px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-semibold font-mono text-xs flex items-center space-x-2 transition shadow-lg shadow-cyan-500/20"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${runDriftMutation.isPending ? "animate-spin" : ""}`} />
          <span>{runDriftMutation.isPending ? "Calculating Drift..." : "Run Live Drift Check"}</span>
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Overall Status</div>
          <div className="text-xl font-bold text-emerald-400 font-mono mt-1">
            {drift?.overall_status || "NO_DRIFT"}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">Evaluated across 15 features</div>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Drifting Features</div>
          <div className="text-xl font-bold text-white font-mono mt-1">
            {drift?.drifting_features_count ?? 0}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">PSI &gt;= 0.25 (Critical)</div>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Features with Warning</div>
          <div className="text-xl font-bold text-amber-400 font-mono mt-1">
            {drift?.warning_features_count ?? 0}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">0.10 &lt;= PSI &lt; 0.25</div>
        </div>

        <div className="glass-card rounded-xl p-5 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Baseline Sample Size</div>
          <div className="text-xl font-bold text-cyan-300 font-mono mt-1">
            {drift?.reference_records?.toLocaleString() || "33,252"}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">Current Stream: {drift?.current_records?.toLocaleString() || "8,299"}</div>
        </div>
      </div>

      {/* Feature-by-Feature Table */}
      <div className="glass-card rounded-2xl p-6 border border-slate-800">
        <h2 className="text-sm font-semibold text-white font-mono uppercase tracking-wider mb-4">
          Feature Distribution Drift Matrix
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="pb-3 px-3">Feature Name</th>
                <th className="pb-3 px-3">Status</th>
                <th className="pb-3 px-3">PSI (Quantile)</th>
                <th className="pb-3 px-3">KS Statistic</th>
                <th className="pb-3 px-3">KS p-value</th>
                <th className="pb-3 px-3">JS Divergence</th>
                <th className="pb-3 px-3 text-right">Reference μ ± σ</th>
                <th className="pb-3 px-3 text-right">Current μ ± σ</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {drift?.features?.map((f) => (
                <tr key={f.feature} className="hover:bg-slate-900/40 text-slate-300">
                  <td className="py-3 px-3 font-semibold text-cyan-300">{f.feature}</td>
                  <td className="py-3 px-3">
                    <span className={`text-[10px] px-2 py-0.5 rounded border uppercase ${getStatusBadge(f.status)}`}>
                      {f.status}
                    </span>
                  </td>
                  <td className="py-3 px-3 font-bold">{f.psi.toFixed(4)}</td>
                  <td className="py-3 px-3 text-slate-300">{f.ks_statistic.toFixed(4)}</td>
                  <td className="py-3 px-3 text-slate-400">{f.ks_pvalue.toFixed(4)}</td>
                  <td className="py-3 px-3 text-slate-300">{f.jensen_shannon_distance.toFixed(4)}</td>
                  <td className="py-3 px-3 text-right text-slate-400">
                    {f.reference_mean.toFixed(2)} ± {f.reference_std.toFixed(2)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {f.current_mean.toFixed(2)} ± {f.current_std.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
