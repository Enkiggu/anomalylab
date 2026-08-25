"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowRight,
  Boxes,
  CheckCircle2,
  Cpu,
  Layers,
  Percent,
  RefreshCw,
  Shield,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import { api, ModelManifest } from "@/lib/api";

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState<ModelManifest | null>(null);
  const [stageFilter, setStageFilter] = useState<string>("all");

  const { data: models = [], isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: api.getModels,
  });

  const promoteMutation = useMutation({
    mutationFn: ({ version, stage }: { version: string; stage: string }) =>
      api.promoteModel(version, stage),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["currentModel"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });

  const filteredModels = models.filter((m) => {
    if (stageFilter === "all") return true;
    return m.stage === stageFilter;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 text-xs font-mono mb-2">
            <Boxes className="h-3.5 w-3.5" />
            <span>MODEL REGISTRY &amp; GOVERNANCE FLEET</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-white font-mono">
            Model Governance &amp; Versioning
          </h1>
          <p className="text-sm text-slate-400">
            Strict policy gates: Production Champion, Candidate Challenger, Rejected, and Archived.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => queryClient.invalidateQueries({ queryKey: ["models"] })}
            className="px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 text-xs text-slate-300 font-mono flex items-center space-x-2 transition"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span>Refresh Fleet</span>
          </button>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-3 text-xs font-mono">
        {["all", "production", "candidate", "rejected", "archived"].map((s) => (
          <button
            key={s}
            onClick={() => setStageFilter(s)}
            className={`px-3 py-1.5 rounded-lg capitalize transition ${
              stageFilter === s
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {s} ({s === "all" ? models.length : models.filter((m) => m.stage === s).length})
          </button>
        ))}
      </div>

      {/* Model Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredModels.map((m) => {
          const isProd = m.stage === "production";
          const isRejected = m.stage === "rejected";
          const metrics = m.metrics || {};
          const valMetrics = m.validation_metrics || {};

          let badgeColor = "bg-slate-900 text-slate-400 border-slate-700";
          if (isProd) badgeColor = "bg-emerald-950 text-emerald-400 border-emerald-500/40";
          else if (m.stage === "candidate") badgeColor = "bg-cyan-950 text-cyan-400 border-cyan-500/40";
          else if (isRejected) badgeColor = "bg-rose-950 text-rose-400 border-rose-500/40";
          else if (m.stage === "experimental") badgeColor = "bg-purple-950 text-purple-400 border-purple-500/40";

          return (
            <div
              key={m.model_version}
              className={`glass-card rounded-xl p-6 border transition-all flex flex-col justify-between ${
                isProd
                  ? "border-emerald-500/50 glow-emerald"
                  : isRejected
                  ? "border-rose-900/40 bg-rose-950/10"
                  : "border-slate-800 hover:border-slate-700"
              }`}
            >
              <div>
                <div className="flex items-start justify-between">
                  <div>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded uppercase font-semibold border ${badgeColor}`}
                    >
                      {m.stage}
                    </span>
                    <h3 className="text-lg font-bold text-white font-mono mt-2">{m.algorithm}</h3>
                  </div>

                  {isProd && <ShieldCheck className="h-6 w-6 text-emerald-400" />}
                  {isRejected && <XCircle className="h-6 w-6 text-rose-400" />}
                </div>

                <div className="mt-2 text-xs font-mono text-cyan-300 truncate" title={m.model_version}>
                  {m.model_version}
                </div>

                {/* Validation Gate Alert for Rejected Models */}
                {isRejected && (
                  <div className="mt-3 p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/40 text-[11px] font-mono text-rose-300 flex items-start space-x-2">
                    <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                    <div>
                      <span>Promotion Gate Failed: FPR ({((valMetrics.false_positive_rate || metrics.false_positive_rate || 0) * 100).toFixed(1)}%) exceeded 5.0% max threshold.</span>
                    </div>
                  </div>
                )}

                {/* Metrics Chips */}
                <div className="mt-5 grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">PR-AUC</div>
                    <div className="text-sm font-bold text-cyan-400">
                      {((metrics.pr_auc || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">Test Recall</div>
                    <div className="text-sm font-bold text-emerald-400">
                      {((metrics.recall || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">Test Precision</div>
                    <div className="text-sm font-bold text-indigo-300">
                      {((metrics.precision || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">FPR / 10k</div>
                    <div className={`text-sm font-bold ${isRejected ? "text-rose-400" : "text-amber-400"}`}>
                      {(metrics.fp_per_10k || 0).toFixed(1)}
                    </div>
                  </div>
                </div>

                {/* Hyperparameters preview */}
                <div className="mt-4 pt-4 border-t border-slate-800/80 text-xs">
                  <div className="text-[10px] text-slate-500 font-mono uppercase mb-1">
                    Dataset &amp; Threshold
                  </div>
                  <div className="text-[11px] font-mono text-slate-400">
                    Threshold: <span className="text-slate-200">{m.threshold.toFixed(4)}</span> • Data: <span className="text-slate-200">{m.dataset_version}</span>
                  </div>
                </div>
              </div>

              {/* Promotion Action */}
              <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between">
                <button
                  onClick={() => setSelectedModel(m)}
                  className="text-xs text-slate-400 hover:text-white font-mono transition"
                >
                  View JSON Manifest →
                </button>

                {!isProd && (
                  <button
                    disabled={isRejected || promoteMutation.isPending}
                    onClick={() =>
                      promoteMutation.mutate({ version: m.model_version, stage: "production" })
                    }
                    title={isRejected ? "Cannot promote model failing validation policy gate" : "Promote to Production Champion"}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition ${
                      isRejected
                        ? "bg-slate-900 text-slate-600 border border-slate-800 cursor-not-allowed"
                        : "bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300"
                    }`}
                  >
                    {isRejected ? "Gate Failed" : "Promote Champion"}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Manifest Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-card max-w-3xl w-full rounded-2xl p-6 border border-slate-700 max-h-[85vh] flex flex-col">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <div>
                <h3 className="text-lg font-bold text-white font-mono">
                  {selectedModel.model_version}
                </h3>
                <p className="text-xs text-slate-400">Model Manifest &amp; Lineage Artifact</p>
              </div>
              <button
                onClick={() => setSelectedModel(null)}
                className="px-3 py-1 rounded-lg bg-slate-800 text-xs font-mono text-slate-300 hover:bg-slate-700"
              >
                Close
              </button>
            </div>

            <pre className="flex-1 overflow-auto mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-cyan-300">
              {JSON.stringify(selectedModel, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
