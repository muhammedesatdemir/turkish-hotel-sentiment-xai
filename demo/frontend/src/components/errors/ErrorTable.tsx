import { ChevronLeft, ChevronRight, Wand2 } from "lucide-react";
import type { ErrorRow } from "@/api/types";
import SentimentBadge from "@/components/prediction/SentimentBadge";
import Button from "@/components/ui/Button";
import { formatPercent } from "@/lib/utils";

const CAT_LABEL: Record<string, string> = {
  "guvenli_yanlis":            "güvenli yanlış",
  "orta_seviye":               "orta seviye",
  "belirsiz_dengesiz":         "belirsiz/dengesiz",
  "etiket_gurultusu_supheli":  "etiket gürültüsü",
};

export default function ErrorTable({
  rows,
  total,
  limit,
  offset,
  onPage,
  onOpen,
}: {
  rows: ErrorRow[];
  total: number;
  limit: number;
  offset: number;
  onPage: (newOffset: number) => void;
  onOpen: (row: ErrorRow) => void;
}) {
  const page = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const showStart = total === 0 ? 0 : offset + 1;
  const showEnd = Math.min(offset + limit, total);

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-md border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Yorum</th>
              <th className="px-3 py-2 text-left font-medium">Gerçek</th>
              <th className="px-3 py-2 text-left font-medium">Tahmin</th>
              <th className="px-3 py-2 text-right font-medium">Güven</th>
              <th className="px-3 py-2 text-left font-medium">Kategori</th>
              <th className="px-3 py-2 text-right font-medium">Aksiyon</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {rows.map((r) => (
              <tr key={r.id} className="hover:bg-gray-50">
                <td className="max-w-md px-3 py-2 text-gray-700">
                  <span className="line-clamp-2" title={r.text}>{r.text}</span>
                  <p className="mt-0.5 text-[11px] text-gray-400">{r.wc} kelime</p>
                </td>
                <td className="px-3 py-2"><SentimentBadge sentiment={r.true} size="sm" /></td>
                <td className="px-3 py-2"><SentimentBadge sentiment={r.pred} size="sm" /></td>
                <td className="px-3 py-2 text-right font-mono text-xs">
                  {formatPercent(r.max_prob, 1)}
                  <p className="mt-0.5 text-[11px] text-gray-400">2nd: {formatPercent(r.second_prob, 1)}</p>
                </td>
                <td className="px-3 py-2 text-xs text-gray-600">{CAT_LABEL[r.category] ?? r.category}</td>
                <td className="px-3 py-2 text-right">
                  <Button variant="secondary" size="sm" onClick={() => onOpen(r)}>
                    <Wand2 className="h-3.5 w-3.5" />
                    XAI
                  </Button>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-3 py-8 text-center text-sm text-gray-500">
                  Filtreye uyan kayıt yok.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between text-xs text-gray-600">
        <span>
          {showStart}-{showEnd} / {total}
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="secondary"
            size="sm"
            disabled={offset === 0}
            onClick={() => onPage(Math.max(0, offset - limit))}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            Önceki
          </Button>
          <span className="px-2 font-mono">
            {page} / {totalPages}
          </span>
          <Button
            variant="secondary"
            size="sm"
            disabled={offset + limit >= total}
            onClick={() => onPage(offset + limit)}
          >
            Sonraki
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
