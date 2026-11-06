import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Search } from "lucide-react";
import type { BatchItem, Sentiment } from "@/api/types";
import { LABELS, SENTIMENT_LABEL } from "@/lib/constants";
import SentimentBadge from "@/components/prediction/SentimentBadge";
import { Input } from "@/components/ui/Input";
import { cn, formatPercent } from "@/lib/utils";

type SortKey = "index" | "sentiment" | "confidence" | "wc";
type SortDir = "asc" | "desc";

export default function BatchResultsTable({ items }: { items: BatchItem[] }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<Sentiment | "all">("all");
  const [sortKey, setSortKey] = useState<SortKey>("index");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const filtered = useMemo(() => {
    const q = search.trim().toLocaleLowerCase("tr-TR");
    return items.filter((it) => {
      if (filter !== "all" && it.sentiment !== filter) return false;
      if (q && !it.text.toLocaleLowerCase("tr-TR").includes(q)) return false;
      return true;
    });
  }, [items, search, filter]);

  const sorted = useMemo(() => {
    const copy = [...filtered];
    copy.sort((a, b) => {
      let av: number | string = a[sortKey];
      let bv: number | string = b[sortKey];
      if (sortKey === "sentiment") {
        av = LABELS.indexOf(a.sentiment);
        bv = LABELS.indexOf(b.sentiment);
      }
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [filtered, sortKey, sortDir]);

  const toggleSort = (k: SortKey) => {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir(k === "confidence" ? "desc" : "asc");
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[200px]">
          <Search aria-hidden className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <Input
            placeholder="Yorumda ara…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="flex gap-1 rounded-md border border-gray-200 bg-white p-0.5">
          {(["all", ...LABELS] as const).map((opt) => (
            <button
              key={opt}
              onClick={() => setFilter(opt)}
              className={cn(
                "rounded-sm px-2.5 py-1 text-xs font-medium transition-colors",
                filter === opt ? "bg-primary text-white" : "text-gray-600 hover:bg-gray-100",
              )}
            >
              {opt === "all" ? "Tümü" : SENTIMENT_LABEL[opt]}
            </button>
          ))}
        </div>
        <span className="text-xs text-gray-500">
          {sorted.length}/{items.length}
        </span>
      </div>

      <div className="overflow-x-auto rounded-md border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
            <tr>
              <SortableTh label="#" k="index" current={sortKey} dir={sortDir} onClick={toggleSort} className="w-12" />
              <th className="px-3 py-2 text-left font-medium">Yorum</th>
              <SortableTh label="Tahmin" k="sentiment" current={sortKey} dir={sortDir} onClick={toggleSort} />
              <SortableTh label="Güven" k="confidence" current={sortKey} dir={sortDir} onClick={toggleSort} className="text-right" />
              <SortableTh label="Kelime" k="wc" current={sortKey} dir={sortDir} onClick={toggleSort} className="text-right" />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {sorted.map((it) => (
              <tr key={it.index} className="hover:bg-gray-50">
                <td className="px-3 py-2 font-mono text-xs text-gray-500">{it.index + 1}</td>
                <td className="max-w-md px-3 py-2 text-gray-700">
                  <span className="line-clamp-2" title={it.text}>{it.text}</span>
                </td>
                <td className="px-3 py-2">
                  <SentimentBadge sentiment={it.sentiment} size="sm" />
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs">
                  {formatPercent(it.confidence, 1)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs text-gray-500">{it.wc}</td>
              </tr>
            ))}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={5} className="px-3 py-8 text-center text-sm text-gray-500">
                  Filtreyle eşleşen sonuç yok.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SortableTh({
  label,
  k,
  current,
  dir,
  onClick,
  className,
}: {
  label: string;
  k: SortKey;
  current: SortKey;
  dir: SortDir;
  onClick: (k: SortKey) => void;
  className?: string;
}) {
  const active = current === k;
  const Icon = !active ? ArrowUpDown : dir === "asc" ? ArrowUp : ArrowDown;
  return (
    <th className={cn("px-3 py-2 text-left font-medium", className)}>
      <button
        onClick={() => onClick(k)}
        className={cn(
          "inline-flex items-center gap-1 rounded px-1 py-0.5 text-xs uppercase tracking-wide",
          active ? "text-gray-900" : "text-gray-500 hover:text-gray-700",
        )}
      >
        <span>{label}</span>
        <Icon className="h-3.5 w-3.5" />
      </button>
    </th>
  );
}
