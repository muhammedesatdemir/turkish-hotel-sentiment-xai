import type { Sentiment } from "@/api/types";
import { SENTIMENT_LABEL } from "@/lib/constants";

export default function TopFeaturesTable({
  featuresByClass,
  predicted,
  topK = 6,
}: {
  featuresByClass: Record<string, [string, number][]>;
  predicted: Sentiment;
  topK?: number;
}) {
  const features = (featuresByClass[predicted] ?? []).slice(0, topK);
  if (features.length === 0) {
    return <p className="text-sm text-gray-500">Bu sınıf için kelime katkısı yok.</p>;
  }
  return (
    <div className="overflow-hidden rounded-md border border-gray-200">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Kelime</th>
            <th className="px-3 py-2 text-right font-medium">
              {SENTIMENT_LABEL[predicted]} katkısı
            </th>
            <th className="px-3 py-2 text-left font-medium">Yön</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {features.map(([word, weight]) => (
            <tr key={word}>
              <td className="px-3 py-2 font-mono">{word}</td>
              <td className={"px-3 py-2 text-right font-mono " + (weight >= 0 ? "text-emerald-700" : "text-red-700")}>
                {weight >= 0 ? "+" : ""}
                {weight.toFixed(4)}
              </td>
              <td className="px-3 py-2 text-gray-500">
                {weight >= 0 ? "destekler" : "karşı çıkar"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
