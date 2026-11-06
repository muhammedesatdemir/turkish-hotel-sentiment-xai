import type { ConfusionMatrixData } from "@/api/types";
import { SENTIMENT_LABEL } from "@/lib/constants";
import { formatPercent } from "@/lib/utils";

export default function ConfusionMatrix({ data }: { data: ConfusionMatrixData }) {
  const { labels, matrix } = data;

  const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0));

  const colorFor = (rowIdx: number, colIdx: number, value: number) => {
    const rowSum = rowSums[rowIdx] || 1;
    const ratio = value / rowSum;
    const isDiagonal = rowIdx === colIdx;
    const r = isDiagonal ? 16 : 239;
    const g = isDiagonal ? 185 : 68;
    const b = isDiagonal ? 129 : 68;
    const alpha = 0.05 + 0.7 * ratio;
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1 text-sm">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left text-xs font-medium uppercase tracking-wide text-gray-500">
              gerçek ↓ / tahmin →
            </th>
            {labels.map((l) => (
              <th key={l} className="px-2 py-1 text-center text-xs font-semibold text-gray-700">
                {SENTIMENT_LABEL[l]}
              </th>
            ))}
            <th className="px-2 py-1 text-center text-xs font-medium uppercase tracking-wide text-gray-500">
              toplam
            </th>
          </tr>
        </thead>
        <tbody>
          {labels.map((trueLabel, i) => (
            <tr key={trueLabel}>
              <th className="px-2 py-1 text-left text-xs font-semibold text-gray-700">
                {SENTIMENT_LABEL[trueLabel]}
              </th>
              {labels.map((_, j) => {
                const value = matrix[i][j];
                const ratio = rowSums[i] ? value / rowSums[i] : 0;
                const isDiagonal = i === j;
                return (
                  <td
                    key={j}
                    className="px-3 py-3 text-center"
                    style={{ backgroundColor: colorFor(i, j, value) }}
                    title={`${trueLabel} → ${labels[j]}: ${value} (${formatPercent(ratio, 1)})`}
                  >
                    <div className={"font-mono text-base " + (isDiagonal ? "font-bold text-emerald-900" : "text-red-900")}>
                      {value}
                    </div>
                    <div className="font-mono text-[10px] text-gray-700">
                      {formatPercent(ratio, 1)}
                    </div>
                  </td>
                );
              })}
              <td className="px-2 py-1 text-center font-mono text-xs text-gray-500">
                {rowSums[i]}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-gray-500">{data.note}</p>
    </div>
  );
}
