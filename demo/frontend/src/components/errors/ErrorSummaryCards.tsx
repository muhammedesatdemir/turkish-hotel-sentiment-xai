import { AlertTriangle, ListFilter, Target } from "lucide-react";
import type { ErrorSummary } from "@/api/types";
import MetricCard from "@/components/dashboard/MetricCard";

const DIR_LABEL: Record<string, string> = {
  "nötr_to_pozitif":     "nötr → pozitif",
  "nötr_to_negatif":     "nötr → negatif",
  "negatif_to_nötr":     "negatif → nötr",
  "pozitif_to_nötr":     "pozitif → nötr",
  "pozitif_to_negatif":  "pozitif → negatif",
};

const CAT_LABEL: Record<string, string> = {
  "guvenli_yanlis":            "güvenli yanlış",
  "orta_seviye":               "orta seviye",
  "belirsiz_dengesiz":         "belirsiz / dengesiz",
  "etiket_gurultusu_supheli":  "etiket gürültüsü şüpheli",
};

export default function ErrorSummaryCards({ data }: { data: ErrorSummary }) {
  const sortedDirs = Object.entries(data.errors_by_direction).sort((a, b) => b[1] - a[1]);
  const sortedCats = Object.entries(data.errors_by_category).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          accent="negatif"
          icon={<AlertTriangle className="h-4 w-4" />}
          label="Toplam Hata"
          value={data.total_errors}
          sublabel={`${data.total_test} test örneği · %${(data.error_rate * 100).toFixed(2)}`}
        />
        <MetricCard
          accent="primary"
          icon={<Target className="h-4 w-4" />}
          label="Ortalama Güven"
          value={data.confidence_stats.mean_max_prob.toFixed(3)}
          sublabel={`medyan ${data.confidence_stats.median_max_prob.toFixed(3)}`}
        />
        <MetricCard
          accent="negatif"
          icon={<ListFilter className="h-4 w-4" />}
          label="Yüksek Güvenli Hata"
          value={data.confidence_stats.high_conf_errors_count}
          sublabel={`%${((data.confidence_stats.high_conf_errors_count / data.total_errors) * 100).toFixed(1)} (≥0.8)`}
        />
        <MetricCard
          accent="notr"
          icon={<ListFilter className="h-4 w-4" />}
          label="Düşük Güvenli Hata"
          value={data.confidence_stats.low_conf_errors_count}
          sublabel="< 0.6"
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Hata yön dağılımı
          </h4>
          <ul className="space-y-1.5">
            {sortedDirs.map(([dir, count]) => {
              const ratio = count / data.total_errors;
              return (
                <li key={dir} className="flex items-center gap-3 text-sm">
                  <span className="w-32 text-xs text-gray-600">{DIR_LABEL[dir] ?? dir}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-100">
                    <div
                      className="h-full rounded-full bg-red-500"
                      style={{ width: `${Math.max(2, ratio * 100)}%`, opacity: 0.4 + 0.6 * ratio }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-xs text-gray-700">{count}</span>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Kategori dağılımı
          </h4>
          <ul className="space-y-1.5">
            {sortedCats.map(([cat, count]) => {
              const ratio = count / data.total_errors;
              return (
                <li key={cat} className="flex items-center gap-3 text-sm">
                  <span className="w-40 text-xs text-gray-600">{CAT_LABEL[cat] ?? cat}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-100">
                    <div
                      className="h-full rounded-full bg-amber-500"
                      style={{ width: `${Math.max(2, ratio * 100)}%`, opacity: 0.4 + 0.6 * ratio }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-xs text-gray-700">{count}</span>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </div>
  );
}
