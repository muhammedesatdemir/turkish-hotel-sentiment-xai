import { Star } from "lucide-react";
import type { MeanStd, MultiModelData } from "@/api/types";
import { cn } from "@/lib/utils";

const COL_LABELS: Record<string, string> = {
  accuracy:    "Accuracy",
  macro_f1:    "Macro-F1",
  weighted_f1: "Weighted-F1",
  f1_negatif:  "F1-negatif",
  f1_notr:     "F1-nötr",
  f1_pozitif:  "F1-pozitif",
};

function fmt(m?: MeanStd): string {
  if (!m) return "—";
  if (m.std === null || m.std === undefined) return m.mean.toFixed(4);
  return `${m.mean.toFixed(4)} ± ${m.std.toFixed(4)}`;
}

export default function MultiModelTable({ data }: { data: MultiModelData }) {
  const sorted = [...data.models].sort(
    (a, b) => (b.metrics.macro_f1?.mean ?? 0) - (a.metrics.macro_f1?.mean ?? 0),
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
          <tr>
            <th className="sticky left-0 z-10 bg-gray-50 px-3 py-2 text-left font-medium">Model</th>
            {data.metric_keys.map((k) => (
              <th key={k} className="px-3 py-2 text-right font-medium">
                {COL_LABELS[k] ?? k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {sorted.map((m) => (
            <tr key={m.key} className={cn(m.highlighted && "bg-primary-50/60")}>
              <td className="sticky left-0 z-10 bg-inherit px-3 py-2">
                <div className="flex items-center gap-2">
                  {m.highlighted && <Star aria-hidden className="h-4 w-4 fill-amber-400 text-amber-500" />}
                  <span className={cn("font-medium", m.highlighted ? "text-primary-900" : "text-gray-900")}>
                    {m.name}
                  </span>
                </div>
              </td>
              {data.metric_keys.map((k) => (
                <td key={k} className="px-3 py-2 text-right font-mono text-xs text-gray-700">
                  {fmt(m.metrics[k])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
