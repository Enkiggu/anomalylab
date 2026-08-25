import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#080c14",
        surface: "#0f1624",
        "surface-raised": "#172136",
        border: "#1e2c47",
        primary: {
          DEFAULT: "#06b6d4", // Cyan
          hover: "#0891b2",
        },
        accent: {
          emerald: "#10b981",
          amber: "#f59e0b",
          rose: "#ef4444",
          indigo: "#6366f1",
          purple: "#a855f7",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
