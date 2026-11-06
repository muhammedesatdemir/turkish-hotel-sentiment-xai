import type { Sentiment } from "@/api/types";
import { SENTIMENT_COLORS } from "@/lib/constants";
import { formatPercent } from "@/lib/utils";

export default function ConfidenceMeter({
  value,
  sentiment,
}: {
  value: number;
  sentiment: Sentiment;
}) {
  const size = 140;
  const stroke = 12;
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = Math.PI * r;
  const clamped = Math.max(0, Math.min(1, value));
  const dashOffset = circumference * (1 - clamped);
  const color = SENTIMENT_COLORS[sentiment];

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size / 2 + stroke} viewBox={`0 0 ${size} ${size / 2 + stroke}`}>
        <path
          d={`M ${stroke / 2} ${cy} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${cy}`}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d={`M ${stroke / 2} ${cy} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          style={{ transition: "stroke-dashoffset 700ms ease-out" }}
        />
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          className="font-mono"
          fontSize="22"
          fontWeight="700"
          fill="#111827"
        >
          {formatPercent(clamped, 1)}
        </text>
      </svg>
      <p className="text-xs uppercase tracking-wider text-gray-500">Güven skoru</p>
    </div>
  );
}
