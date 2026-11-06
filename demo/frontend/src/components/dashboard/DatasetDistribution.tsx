import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import type { DatasetDistribution as DatasetData } from "@/api/types";
import { SENTIMENT_COLORS } from "@/lib/constants";
import type { Sentiment } from "@/api/types";

const SOURCE_COLORS: Record<string, string> = {
  otelpuan_otelyorum: "#1e40af",
  ai_generated:       "#8b5cf6",
  manuel_toplama:     "#f59e0b",
};

const SOURCE_LABELS: Record<string, string> = {
  otelpuan_otelyorum: "otelpuan_otelyorum",
  ai_generated:       "ai_generated",
  manuel_toplama:     "manuel_toplama",
};

const CLASS_LABEL_TR: Record<string, string> = {
  pozitif: "Pozitif",
  "nötr":  "Nötr",
  negatif: "Negatif",
};

export default function DatasetDistribution({ data }: { data: DatasetData }) {
  const total = data.total;
  const classRows = Object.entries(data.by_class).map(([name, value]) => ({
    name,
    value,
    pct: (value / total) * 100,
  }));
  const sourceRows = Object.entries(data.by_source).map(([name, value]) => ({
    name,
    value,
    pct: (value / total) * 100,
  }));
  const descs = (data as DatasetData & { source_descriptions?: Record<string, string> })
    .source_descriptions;

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Sınıf dağılımı */}
      <div>
        <h4 className="mb-2 text-sm font-semibold text-gray-700">
          Sınıf dağılımı
        </h4>
        <div className="h-56">
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={classRows}
                dataKey="value"
                nameKey="name"
                innerRadius={45}
                outerRadius={80}
                paddingAngle={2}
                isAnimationActive
              >
                {classRows.map((row) => (
                  <Cell
                    key={row.name}
                    fill={SENTIMENT_COLORS[row.name as Sentiment] ?? "#9ca3af"}
                  />
                ))}
              </Pie>
              <Tooltip
                formatter={(v: number, _name: string, item: { payload?: { pct?: number } }) => {
                  const pct = item.payload?.pct;
                  return [
                    `${v.toLocaleString("tr-TR")}${
                      pct !== undefined ? ` (%${pct.toFixed(1)})` : ""
                    }`,
                    "yorum",
                  ];
                }}
                contentStyle={{
                  borderRadius: 8,
                  fontSize: 12,
                  border: "1px solid #e5e7eb",
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 overflow-hidden rounded-md border border-gray-200">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="px-2 py-1 text-left font-medium">Sınıf</th>
                <th className="px-2 py-1 text-right font-medium">Sayı</th>
                <th className="px-2 py-1 text-right font-medium">Oran</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {classRows.map((row) => (
                <tr key={row.name}>
                  <td className="px-2 py-1 text-gray-800">
                    {CLASS_LABEL_TR[row.name] ?? row.name}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-gray-700">
                    {row.value.toLocaleString("tr-TR")}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-gray-700">
                    %{row.pct.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Kaynak dağılımı */}
      <div>
        <h4 className="mb-2 text-sm font-semibold text-gray-700">
          Kaynak dağılımı
        </h4>
        <div className="h-56">
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={sourceRows}
                dataKey="value"
                nameKey="name"
                innerRadius={45}
                outerRadius={80}
                paddingAngle={2}
                isAnimationActive
              >
                {sourceRows.map((row) => (
                  <Cell key={row.name} fill={SOURCE_COLORS[row.name] ?? "#9ca3af"} />
                ))}
              </Pie>
              <Tooltip
                formatter={(v: number, _name: string, item: { payload?: { pct?: number } }) => {
                  const pct = item.payload?.pct;
                  return [
                    `${v.toLocaleString("tr-TR")}${
                      pct !== undefined ? ` (%${pct.toFixed(1)})` : ""
                    }`,
                    "yorum",
                  ];
                }}
                contentStyle={{
                  borderRadius: 8,
                  fontSize: 12,
                  border: "1px solid #e5e7eb",
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: 12 }}
                formatter={(label) =>
                  SOURCE_LABELS[label as string] ?? (label as string)
                }
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 overflow-hidden rounded-md border border-gray-200">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="px-2 py-1 text-left font-medium">Kaynak</th>
                <th className="px-2 py-1 text-right font-medium">Sayı</th>
                <th className="px-2 py-1 text-right font-medium">Oran</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {sourceRows.map((row) => (
                <tr key={row.name}>
                  <td className="px-2 py-1 text-gray-800">
                    <span className="font-mono">{row.name}</span>
                    {descs?.[row.name] && (
                      <span className="ml-1 text-[10px] text-gray-500">
                        — {descs[row.name]}
                      </span>
                    )}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-gray-700">
                    {row.value.toLocaleString("tr-TR")}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-gray-700">
                    %{row.pct.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Train/Val/Test */}
      <div className="md:col-span-2">
        <h4 className="mb-2 text-sm font-semibold text-gray-700">
          Train / Val / Test bölünmesi
        </h4>
        <div className="space-y-2 text-sm">
          {(["train", "val", "test"] as const).map((k) => {
            const v = data.splits[k];
            const ratio = v / data.total;
            return (
              <div key={k}>
                <div className="mb-1 flex justify-between font-mono text-xs">
                  <span className="uppercase tracking-wider text-gray-600">{k}</span>
                  <span className="text-gray-700">
                    {v.toLocaleString("tr-TR")} (%{(ratio * 100).toFixed(1)})
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-gray-100">
                  <div
                    className="h-full rounded-full bg-primary-600"
                    style={{ width: `${(ratio * 100).toFixed(2)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
        <p className="mt-3 rounded-md bg-gray-50 p-2 text-xs text-gray-600">
          <span className="font-semibold">Test seti:</span> %100 gerçek
          kullanıcı yorumu (AI üretimli örnekler test'e sızdırılmadı —{" "}
          <code className="font-mono">--ai_only_in_train</code> bayrağı). Per-class
          test bölünmesi: nötr için %20, diğer sınıflar için %10.
        </p>
      </div>
    </div>
  );
}
