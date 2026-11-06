import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PerClassMetrics } from "@/api/types";
import { LABELS, SENTIMENT_LABEL } from "@/lib/constants";

export default function PerClassChart({ data }: { data: PerClassMetrics }) {
  const rows = LABELS.map((label) => ({
    name: SENTIMENT_LABEL[label],
    Precision: +(data.precision[label] ?? 0).toFixed(4),
    Recall:    +(data.recall[label]    ?? 0).toFixed(4),
    "F1":      +(data.f1[label]        ?? 0).toFixed(4),
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
          <YAxis domain={[0, 1]} stroke="#6b7280" fontSize={12} tickFormatter={(v) => v.toFixed(2)} />
          <Tooltip
            formatter={(value: number) => value.toFixed(4)}
            contentStyle={{ borderRadius: 8, fontSize: 12, border: "1px solid #e5e7eb" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="Precision" fill="#1e40af" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Recall"    fill="#3b82f6" radius={[4, 4, 0, 0]} />
          <Bar dataKey="F1"        fill="#10b981" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
