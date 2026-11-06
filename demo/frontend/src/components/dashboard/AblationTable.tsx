import type { AblationData, MeanStd } from "@/api/types";
import { cn } from "@/lib/utils";

const COL_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  macro_f1: "Macro-F1",
  f1_notr:  "F1-nötr",
};

function fmt(m?: MeanStd): string {
  if (!m) return "—";
  if (m.std === null || m.std === undefined) return m.mean.toFixed(4);
  return `${m.mean.toFixed(4)} ± ${m.std.toFixed(4)}`;
}

export default function AblationTable({ data }: { data: AblationData }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Aşama</th>
            <th className="px-3 py-2 text-left font-medium">Açıklama</th>
            <th className="px-3 py-2 text-center font-medium">Seed</th>
            {data.metric_keys.map((k) => (
              <th key={k} className="px-3 py-2 text-right font-medium">
                {COL_LABELS[k] ?? k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {data.stages.map((s) => (
            <tr key={s.key} className={cn(s.highlighted && "bg-emerald-50/60")}>
              <td className="px-3 py-2">
                <span className={cn("font-medium", s.highlighted ? "text-emerald-900" : "text-gray-900")}>
                  {s.label}
                </span>
                {s.highlighted && (
                  <span className="ml-2 inline-flex items-center rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-800">
                    nihai
                  </span>
                )}
              </td>
              <td className="px-3 py-2 text-gray-600">
                {s.description}
                {s.note && <p className="mt-0.5 text-xs italic text-gray-500">{s.note}</p>}
              </td>
              <td className="px-3 py-2 text-center font-mono text-xs text-gray-500">
                {s.n_seeds}×
              </td>
              {data.metric_keys.map((k) => (
                <td key={k} className="px-3 py-2 text-right font-mono text-xs text-gray-700">
                  {fmt(s.metrics[k])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
