"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Boxes,
  Database,
  FlaskConical,
  Gauge,
  GitBranch,
  PlaySquare,
  ShieldAlert,
} from "lucide-react";
import { clsx } from "clsx";

const navItems = [
  { name: "Overview", href: "/", icon: Gauge },
  { name: "Model Registry", href: "/models", icon: Boxes },
  { name: "Experiments", href: "/experiments", icon: FlaskConical },
  { name: "Inference Playground", href: "/predictions", icon: PlaySquare },
  { name: "Drift Monitoring", href: "/drift", icon: GitBranch },
  { name: "Dataset Catalog", href: "/datasets", icon: Database },
  { name: "System Metrics", href: "/system", icon: BarChart3 },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border bg-surface/50 p-4 flex flex-col justify-between hidden md:flex min-h-[calc(100vh-4rem)]">
      <nav className="space-y-1">
        <div className="px-3 py-2 text-[10px] font-mono font-semibold uppercase tracking-wider text-slate-500">
          Analytics & MLOps
        </div>
        {navItems.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={clsx(
                "flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all duration-150",
                isActive
                  ? "bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 glow-cyan"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent"
              )}
            >
              <Icon className={clsx("h-4 w-4", isActive ? "text-cyan-400" : "text-slate-400")} />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs">
        <div className="flex items-center space-x-2 text-slate-300 font-mono font-semibold">
          <ShieldAlert className="h-4 w-4 text-cyan-400" />
          <span>Evaluation Engine</span>
        </div>
        <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
          Supervised metrics computed purely on held-out temporal partitions.
        </p>
      </div>
    </aside>
  );
}
