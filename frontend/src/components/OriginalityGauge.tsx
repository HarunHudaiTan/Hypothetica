import { useEffect, useState } from "react";

interface Props {
  score: number;
}

const CX = 100;
const CY = 86;
const R = 66;
const STROKE = 12;

// 270° sweep: bottom-left → top → bottom-right
const START_ANGLE = 225;
const END_ANGLE = 495;
const SPAN = END_ANGLE - START_ANGLE;

const ZONES: { from: number; to: number; color: string }[] = [
  { from: 0, to: 40, color: "#ef4444" },
  { from: 40, to: 70, color: "#eab308" },
  { from: 70, to: 100, color: "#22c55e" },
];

function toRad(deg: number) {
  return ((deg - 90) * Math.PI) / 180;
}

function pt(r: number, deg: number) {
  const rad = toRad(deg);
  return { x: CX + r * Math.cos(rad), y: CY + r * Math.sin(rad) };
}

function arcD(r: number, from: number, to: number) {
  const s = pt(r, from);
  const e = pt(r, to);
  const large = to - from > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
}

function valToAngle(v: number) {
  return START_ANGLE + (Math.max(0, Math.min(100, v)) / 100) * SPAN;
}

export default function OriginalityGauge({ score }: Props) {
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

  const angle = valToAngle(display);
  const tip = pt(R - 14, angle);
  const b1 = pt(4.5, angle - 90);
  const b2 = pt(4.5, angle + 90);

  let accent: string, label: string, pill: string;
  if (score >= 70) {
    accent = "#22c55e";
    label = "High Originality";
    pill = "bg-green-500";
  } else if (score >= 40) {
    accent = "#eab308";
    label = "Moderate Originality";
    pill = "bg-amber-500";
  } else {
    accent = "#ef4444";
    label = "Low Originality";
    pill = "bg-red-500";
  }

  return (
    <div className="flex flex-col items-center py-2">
      <svg viewBox="0 0 200 162" className="w-full max-w-[280px]">
        {/* Background track */}
        <path
          d={arcD(R, START_ANGLE, END_ANGLE)}
          fill="none"
          stroke="#f1f5f9"
          strokeWidth={STROKE}
          strokeLinecap="round"
        />

        {/* Zone colors (full opacity, sits on track) */}
        {ZONES.map((z) => (
          <path
            key={z.from}
            d={arcD(R, valToAngle(z.from), valToAngle(z.to))}
            fill="none"
            stroke={z.color}
            strokeWidth={STROKE}
            strokeLinecap="round"
            opacity={0.15}
          />
        ))}

        {/* Active fill — same width as track, no filter */}
        {display > 0 && (
          <path
            d={arcD(R, START_ANGLE, angle)}
            fill="none"
            stroke={accent}
            strokeWidth={STROKE}
            strokeLinecap="round"
            opacity={0.85}
          />
        )}

        {/* Minor ticks */}
        {Array.from({ length: 21 }, (_, i) => i * 5).map((v) => {
          const a = valToAngle(v);
          const major = v % 20 === 0;
          const o = pt(R + STROKE / 2 + 2, a);
          const inner = pt(R + STROKE / 2 - (major ? 6 : 3), a);
          return (
            <line
              key={v}
              x1={inner.x} y1={inner.y} x2={o.x} y2={o.y}
              stroke={major ? "#94a3b8" : "#cbd5e1"}
              strokeWidth={major ? 1.2 : 0.6}
            />
          );
        })}

        {/* Labels */}
        {[0, 20, 40, 60, 80, 100].map((v) => {
          const p = pt(R + STROKE / 2 + 12, valToAngle(v));
          return (
            <text
              key={v}
              x={p.x} y={p.y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="7.5"
              fontWeight="600"
              className="fill-slate-400"
            >
              {v}
            </text>
          );
        })}

        {/* Needle */}
        <polygon
          points={`${tip.x},${tip.y} ${b1.x},${b1.y} ${b2.x},${b2.y}`}
          fill="#1e293b"
        />
        {/* Needle dot */}
        <circle cx={CX} cy={CY} r="6" fill="#1e293b" />
        <circle cx={CX} cy={CY} r="3" fill="#64748b" />

        {/* Score */}
        <text
          x={CX} y={CY + 26}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#1e293b"
          fontSize="26"
          fontWeight="800"
          fontFamily="system-ui, sans-serif"
        >
          {display}
        </text>
        <text
          x={CX} y={CY + 42}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#94a3b8"
          fontSize="9"
        >
          / 100
        </text>
      </svg>

      <div className={`-mt-2 px-4 py-1.5 rounded-full text-white text-xs font-semibold ${pill}`}>
        {label}
      </div>
    </div>
  );
}
