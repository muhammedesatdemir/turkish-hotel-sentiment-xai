import { useMemo } from "react";
import type { Sentiment } from "@/api/types";

const POS_RGB = "16, 185, 129";   // emerald
const NEG_RGB = "239, 68, 68";    // red

interface Props {
  text: string;
  features: [string, number][];
  predicted: Sentiment;
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
}

export default function LIMEHighlighter({ text, features, predicted }: Props) {
  const { tokens, weights, maxAbs } = useMemo(() => {
    const lookup = new Map<string, number>();
    for (const [word, weight] of features) {
      const key = normalize(word);
      if (key) {
        const prev = lookup.get(key);
        if (prev === undefined || Math.abs(weight) > Math.abs(prev)) {
          lookup.set(key, weight);
        }
      }
    }
    const parts = text.split(/(\s+)/);
    const w: (number | null)[] = parts.map((part) => {
      if (/^\s+$/.test(part)) return null;
      const key = normalize(part);
      return key && lookup.has(key) ? lookup.get(key)! : null;
    });
    const max = w.reduce<number>((m, v) => (v !== null ? Math.max(m, Math.abs(v)) : m), 0);
    return { tokens: parts, weights: w, maxAbs: max || 1 };
  }, [text, features]);

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        Tahmin edilen sınıf <b>{predicted}</b> için kelime katkıları —
        <span className="ml-1 inline-flex items-center gap-1">
          <span className="inline-block h-2 w-3 rounded" style={{ backgroundColor: `rgb(${POS_RGB})` }} />
          <span>destekleyen</span>
        </span>
        <span className="ml-2 inline-flex items-center gap-1">
          <span className="inline-block h-2 w-3 rounded" style={{ backgroundColor: `rgb(${NEG_RGB})` }} />
          <span>karşı çıkan</span>
        </span>
      </p>
      <div className="rounded-md border border-gray-200 bg-gray-50 p-4 text-base leading-relaxed">
        {tokens.map((tok, i) => {
          const w = weights[i];
          if (w === null) return <span key={i}>{tok}</span>;
          const intensity = Math.min(1, Math.abs(w) / maxAbs);
          const alpha = 0.15 + 0.55 * intensity;
          const bg = w >= 0 ? `rgba(${POS_RGB}, ${alpha.toFixed(3)})` : `rgba(${NEG_RGB}, ${alpha.toFixed(3)})`;
          return (
            <span
              key={i}
              title={`${tok}: ${w >= 0 ? "+" : ""}${w.toFixed(4)}`}
              className="rounded px-0.5 py-0.5"
              style={{ backgroundColor: bg }}
            >
              {tok}
            </span>
          );
        })}
      </div>
    </div>
  );
}
