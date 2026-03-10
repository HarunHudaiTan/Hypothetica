import { useEffect, useState } from "react";

interface Props {
  score: number;
  summary?: string;
}

/**
 * OriginalityGauge - Displays the originality score with an animated gradient bar.
 * Shows a red-to-green gradient indicating the originality level (0-100).
 */
export default function OriginalityGauge({ score, summary }: Props) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    let raf: number;
    const t0 = performance.now();
    const ms = 1200;
    const step = (now: number) => {
      const p = Math.min((now - t0) / ms, 1);
      const ease = 1 - (1 - p) * (1 - p) * (1 - p);
      setDisplay(Math.round(ease * score));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [score]);

  // Determine label and colors based on score
  let label: string, textColor: string;
  if (score >= 70) {
    label = "High Originality";
    textColor = "text-green-600";
  } else if (score >= 40) {
    label = "Moderate Originality";
    textColor = "text-amber-600";
  } else {
    label = "Low Originality";
    textColor = "text-red-600";
  }

  return (
    <div className="flex flex-col items-center w-full">
      {/* Summary Text */}
      {summary && (
        <div className="w-full mb-6">
          <p className="text-sm text-slate-600 leading-relaxed text-center italic">
            "{summary}"
          </p>
        </div>
      )}

      {/* Gradient Bar Container */}
      <div className="w-full max-w-md px-4">
        {/* Labels above bar */}
        <div className="flex justify-between text-xs text-slate-400 mb-2">
          <span>Low</span>
          <span>Moderate</span>
          <span>High</span>
        </div>

        {/* Gradient Bar */}
        <div className="relative h-6 rounded-full overflow-hidden shadow-inner">
          {/* Gradient background */}
          <div
            className="absolute inset-0 rounded-full"
            style={{
              background: "linear-gradient(to right, #ef4444 0%, #f59e0b 40%, #eab308 50%, #84cc16 70%, #22c55e 100%)",
            }}
          />

          {/* Position indicator */}
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 transition-all duration-1000 ease-out"
            style={{ left: `${display}%` }}
          >
            {/* Vertical line marker */}
            <div className="w-1 h-8 bg-slate-800 rounded-full shadow-lg border-2 border-white" />
          </div>

          {/* Tick marks */}
          <div className="absolute inset-0 flex justify-between items-center px-0">
            {[0, 20, 40, 60, 80, 100].map((tick) => (
              <div
                key={tick}
                className="w-px h-3 bg-white/40"
                style={{ marginLeft: tick === 0 ? "0" : undefined }}
              />
            ))}
          </div>
        </div>

        {/* Scale numbers below bar */}
        <div className="flex justify-between text-xs text-slate-400 mt-1.5 px-0">
          <span>0</span>
          <span>20</span>
          <span>40</span>
          <span>60</span>
          <span>80</span>
          <span>100</span>
        </div>
      </div>

      {/* Score Display */}
      <div className="mt-6 text-center">
        <div className={`text-5xl font-bold ${textColor}`}>
          {display}
        </div>
        <div className="text-slate-400 text-sm mt-1">
          out of 100
        </div>
        <div className={`mt-2 inline-block px-4 py-1.5 rounded-full text-white text-xs font-semibold ${
          score >= 70 ? "bg-green-500" : score >= 40 ? "bg-amber-500" : "bg-red-500"
        }`}>
          {label}
        </div>
      </div>
    </div>
  );
}
