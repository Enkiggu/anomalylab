"use client";

import { useMutation } from "@tanstack/react-query";
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Layers,
  Play,
  RotateCcw,
  Sparkles,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { api, PredictResponse } from "@/lib/api";

const PRESET_SCENARIOS = [
  {
    name: "Normal Business Login",
    desc: "Legitimate enterprise employee logging in during peak office hours.",
    payload: {
      userId: "user_0042.corp",
      loginHour: 14,
      eventOutcome: "SUCCESS",
      failedAttempts: 0,
      requestCount: 3,
      bytesSent: 1500,
      bytesReceived: 6200,
      sessionDuration: 240,
      isPrivilegedUser: false,
      newSourceIp: false,
      newDevice: false,
    },
  },
  {
    name: "Brute-Force Attack",
    desc: "Rapid authentication failures from untrusted external IP subnet.",
    payload: {
      userId: "user_0019.corp",
      loginHour: 3,
      eventOutcome: "FAILURE",
      failedAttempts: 28,
      requestCount: 85,
      bytesSent: 4200,
      bytesReceived: 1200,
      sessionDuration: 12,
      isPrivilegedUser: true,
      newSourceIp: true,
      newDevice: true,
    },
  },
  {
    name: "Off-Hours Exfiltration",
    desc: "Midnight session transferring abnormal outbound byte volume.",
    payload: {
      userId: "user_0088.corp",
      loginHour: 1,
      eventOutcome: "SUCCESS",
      failedAttempts: 1,
      requestCount: 40,
      bytesSent: 950000,
      bytesReceived: 15000,
      sessionDuration: 850,
      isPrivilegedUser: true,
      newSourceIp: true,
      newDevice: false,
    },
  },
];

export default function PredictionsPage() {
  const [formData, setFormData] = useState(PRESET_SCENARIOS[0].payload);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const predictMutation = useMutation({
    mutationFn: (data: typeof formData) => api.predict(data),
    onSuccess: (data) => setResult(data),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    predictMutation.mutate(formData);
  };

  const getRiskColor = (band: string) => {
    switch (band) {
      case "CRITICAL":
        return "text-rose-400 bg-rose-950/80 border-rose-500/50 glow-rose";
      case "HIGH":
        return "text-amber-400 bg-amber-950/80 border-amber-500/50 glow-amber";
      case "MEDIUM":
        return "text-yellow-300 bg-yellow-950/80 border-yellow-500/40";
      case "LOW":
        return "text-cyan-300 bg-cyan-950/80 border-cyan-500/40";
      default:
        return "text-emerald-400 bg-emerald-950/80 border-emerald-500/40 glow-emerald";
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 text-xs font-mono mb-2">
          <Zap className="h-3.5 w-3.5" />
          <span>REAL-TIME INFERENCE &amp; EXPLAINABILITY</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-white font-mono">
          Online Anomaly Scoring Studio
        </h1>
        <p className="text-sm text-slate-400">
          Simulate incoming security events with sub-5ms latency and non-causal feature deviation breakdowns.
        </p>
      </div>

      {/* Preset Scenarios */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {PRESET_SCENARIOS.map((preset) => (
          <button
            key={preset.name}
            onClick={() => {
              setFormData(preset.payload);
              predictMutation.mutate(preset.payload);
            }}
            className="p-4 rounded-xl glass-card border border-slate-800 hover:border-cyan-500/40 text-left transition"
          >
            <div className="flex items-center justify-between text-xs font-mono text-cyan-400 font-semibold mb-1">
              <span>{preset.name}</span>
              <Play className="h-3 w-3" />
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">{preset.desc}</p>
          </button>
        ))}
      </div>

      {/* Main Grid: Form on Left, Output on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Form Controls */}
        <form onSubmit={handleSubmit} className="lg:col-span-7 space-y-5 glass-card rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h2 className="text-sm font-semibold text-white font-mono uppercase">Event Telemetry Signals</h2>
            <button
              type="button"
              onClick={() => setFormData(PRESET_SCENARIOS[0].payload)}
              className="text-xs text-slate-500 hover:text-slate-300 font-mono flex items-center space-x-1"
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset</span>
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">User Identifier</label>
              <input
                type="text"
                value={formData.userId}
                onChange={(e) => setFormData({ ...formData, userId: e.target.value })}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">Login Hour (0 - 23 UTC)</label>
              <input
                type="number"
                min="0"
                max="23"
                value={formData.loginHour}
                onChange={(e) => setFormData({ ...formData, loginHour: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">Failed Attempts Count</label>
              <input
                type="number"
                min="0"
                value={formData.failedAttempts}
                onChange={(e) => setFormData({ ...formData, failedAttempts: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">5-Minute Request Burst Count</label>
              <input
                type="number"
                min="1"
                value={formData.requestCount}
                onChange={(e) => setFormData({ ...formData, requestCount: parseInt(e.target.value) || 1 })}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">Bytes Sent (Outbound)</label>
              <input
                type="number"
                min="0"
                value={formData.bytesSent}
                onChange={(e) => setFormData({ ...formData, bytesSent: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">Session Duration (Seconds)</label>
              <input
                type="number"
                min="0"
                value={formData.sessionDuration}
                onChange={(e) => setFormData({ ...formData, sessionDuration: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>
          </div>

          {/* Toggle Switches */}
          <div className="pt-3 border-t border-slate-800/80 grid grid-cols-1 sm:grid-cols-3 gap-3">
            <label className="flex items-center space-x-2.5 cursor-pointer text-xs font-mono text-slate-300">
              <input
                type="checkbox"
                checked={formData.newSourceIp}
                onChange={(e) => setFormData({ ...formData, newSourceIp: e.target.checked })}
                className="rounded bg-slate-950 border-slate-700 text-cyan-500 focus:ring-0"
              />
              <span>New Source IP</span>
            </label>

            <label className="flex items-center space-x-2.5 cursor-pointer text-xs font-mono text-slate-300">
              <input
                type="checkbox"
                checked={formData.newDevice}
                onChange={(e) => setFormData({ ...formData, newDevice: e.target.checked })}
                className="rounded bg-slate-950 border-slate-700 text-cyan-500 focus:ring-0"
              />
              <span>Unseen Device</span>
            </label>

            <label className="flex items-center space-x-2.5 cursor-pointer text-xs font-mono text-slate-300">
              <input
                type="checkbox"
                checked={formData.isPrivilegedUser}
                onChange={(e) => setFormData({ ...formData, isPrivilegedUser: e.target.checked })}
                className="rounded bg-slate-950 border-slate-700 text-cyan-500 focus:ring-0"
              />
              <span>Privileged Admin</span>
            </label>
          </div>

          <button
            type="submit"
            disabled={predictMutation.isPending}
            className="w-full py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold font-mono text-xs flex items-center justify-center space-x-2 transition shadow-lg shadow-cyan-500/20"
          >
            <Zap className="h-4 w-4" />
            <span>{predictMutation.isPending ? "Calculating Deviations..." : "Execute Real-Time Scoring"}</span>
          </button>
        </form>

        {/* Prediction Results & Explainability */}
        <div className="lg:col-span-5 space-y-6">
          {result ? (
            <div className="space-y-6">
              {/* Score Gauge Card */}
              <div className="glass-card rounded-2xl p-6 border border-slate-800 relative overflow-hidden">
                <div className="flex items-start justify-between">
                  <div>
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                      Anomaly Score
                    </span>
                    <div className="text-4xl font-bold font-mono text-white mt-1">
                      {result.anomalyScore.toFixed(4)}
                    </div>
                    <div className="text-[11px] font-mono text-slate-400 mt-1">
                      Threshold: <span className="text-cyan-400">{result.threshold.toFixed(4)}</span> • Latency: <span className="text-emerald-400">{result.latencyMs}ms</span>
                    </div>
                  </div>

                  <span
                    className={`px-3 py-1 rounded-full text-xs font-mono font-bold uppercase border ${getRiskColor(
                      result.riskBand
                    )}`}
                  >
                    {result.riskBand} RISK
                  </span>
                </div>

                {/* Score Progress Bar */}
                <div className="w-full bg-slate-950 rounded-full h-3 mt-5 p-0.5 border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      result.isAnomaly ? "bg-rose-500" : "bg-cyan-500"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(5, result.anomalyScore * 100))}%` }}
                  />
                </div>
              </div>

              {/* Contributing Signals Breakdown */}
              <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
                <div className="flex items-center space-x-2">
                  <Sparkles className="h-4 w-4 text-cyan-400" />
                  <h3 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                    Contributing Signals ({result.contributingSignals.length})
                  </h3>
                </div>

                {result.contributingSignals.length > 0 ? (
                  <div className="space-y-3">
                    {result.contributingSignals.map((signal, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs space-y-1.5"
                      >
                        <div className="flex items-center justify-between font-mono">
                          <span className="font-semibold text-cyan-300">{signal.feature}</span>
                          <span className="text-amber-400 font-bold">
                            {signal.deviation > 0 ? `+${signal.deviation.toFixed(1)}σ` : `${signal.deviation.toFixed(1)}σ`}
                          </span>
                        </div>
                        <p className="text-slate-300 text-[11px] leading-relaxed">{signal.message}</p>
                        <div className="text-[10px] text-slate-500 font-mono">
                          Observed: {signal.observedValue} • Baseline: {signal.baselineValue}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800/50 text-center text-xs text-slate-400 font-mono">
                    <CheckCircle2 className="h-5 w-5 text-emerald-400 mx-auto mb-2" />
                    All observed features fall within normal baseline limits.
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="glass-card rounded-2xl p-12 border border-slate-800 text-center space-y-3">
              <Cpu className="h-8 w-8 text-cyan-400 mx-auto" />
              <h3 className="text-sm font-bold font-mono text-white">Inference Engine Ready</h3>
              <p className="text-xs text-slate-400 max-w-xs mx-auto">
                Submit an event payload or select a scenario above to test real-time detection.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
