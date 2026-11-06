import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { BatchItem, Sentiment } from "@/api/types";
import { LABELS, SENTIMENT_COLORS, SENTIMENT_LABEL } from "@/lib/constants";

const HIST_BUCKETS = [
  { label: "<60%",  min: 0.0, max: 0.6 },
  { label: "60-70", min: 0.6, max: 0.7 },
  { label: "70-80", min: 0.7, max: 0.8 },
  { label: "80-90", min: 0.8, max: 0.9 },
  { label: "90-95", min: 0.9, max: 0.95 },
  { label: "≥95%",  min: 0.95, max: 1.01 },
];

export default function BatchDistributionChart({
  distribution,
  items,
}: {
  distribution: Record<string, number>;
  items: BatchItem[];
}) {
  const pieData = LABELS.map((label) => ({
    name: SENTIMENT_LABEL[label],
    key:  label as Sentiment,
    value: distribution[label] ?? 0,
  })).filter((row) => row.value > 0);

  const histData = HIST_BUCKETS.map((b) => ({
    bucket: b.label,
    count: items.filter((it) => it.confidence >= b.min && it.confidence < b.max).length,
  }));

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <h4 className="mb-2 text-sm font-semibold text-gray-700">Sınıf dağılımı</h4>
        <div className="h-56">
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                innerRadius={45}
                outerRadius={80}
                paddingAngle={2}
                label={({ value }) => value}
                isAnimationActive
              >
                {pieData.map((row) => (
                  <Cell key={row.key} fill={SENTIMENT_COLORS[row.key]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(v: number) => v}
                contentStyle={{ borderRadius: 8, fontSize: 12, border: "1px solid #e5e7eb" }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <h4 className="mb-2 text-sm font-semibold text-gray-700">Güven histogramı</h4>
        <div className="h-56">
          <ResponsiveContainer>
            <BarChart data={histData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="bucket" stroke="#6b7280" fontSize={11} />
              <YAxis stroke="#6b7280" fontSize={11} allowDecimals={false} />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: "1px solid #e5e7eb" }}
              />
              <Bar dataKey="count" fill="#1e40af" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
