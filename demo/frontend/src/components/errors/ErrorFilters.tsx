import { ChevronDown, X } from "lucide-react";
import { cn } from "@/lib/utils";

const DIR_OPTIONS = [
  { value: "all",                  label: "Tüm yönler" },
  { value: "nötr_to_pozitif",      label: "nötr → pozitif" },
  { value: "nötr_to_negatif",      label: "nötr → negatif" },
  { value: "negatif_to_nötr",      label: "negatif → nötr" },
  { value: "pozitif_to_nötr",      label: "pozitif → nötr" },
  { value: "pozitif_to_negatif",   label: "pozitif → negatif" },
];

const CAT_OPTIONS = [
  { value: "all",                       label: "Tüm kategoriler" },
  { value: "guvenli_yanlis",            label: "güvenli yanlış" },
  { value: "orta_seviye",               label: "orta seviye" },
  { value: "belirsiz_dengesiz",         label: "belirsiz / dengesiz" },
  { value: "etiket_gurultusu_supheli",  label: "etiket gürültüsü" },
];

export interface FilterState {
  direction: string;
  category: string;
  minConf: number;
  maxConf: number;
}

export const DEFAULT_FILTERS: FilterState = {
  direction: "all",
  category: "all",
  minConf: 0.0,
  maxConf: 1.0,
};

export default function ErrorFilters({
  value,
  onChange,
}: {
  value: FilterState;
  onChange: (v: FilterState) => void;
}) {
  const hasFilter =
    value.direction !== "all" ||
    value.category !== "all" ||
    value.minConf > 0 ||
    value.maxConf < 1;

  return (
    <div className="flex flex-wrap items-end gap-3">
      <SelectField
        label="Yön"
        value={value.direction}
        onChange={(v) => onChange({ ...value, direction: v })}
        options={DIR_OPTIONS}
      />
      <SelectField
        label="Kategori"
        value={value.category}
        onChange={(v) => onChange({ ...value, category: v })}
        options={CAT_OPTIONS}
      />
      <div>
        <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-gray-500">
          Güven aralığı
        </label>
        <div className="flex items-center gap-2 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm">
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={value.minConf}
            onChange={(e) => onChange({ ...value, minConf: clamp(Number(e.target.value), 0, value.maxConf) })}
            className="w-16 bg-transparent font-mono text-xs focus:outline-none"
          />
          <span className="text-xs text-gray-400">—</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={value.maxConf}
            onChange={(e) => onChange({ ...value, maxConf: clamp(Number(e.target.value), value.minConf, 1) })}
            className="w-16 bg-transparent font-mono text-xs focus:outline-none"
          />
        </div>
      </div>
      {hasFilter && (
        <button
          onClick={() => onChange(DEFAULT_FILTERS)}
          className="inline-flex h-9 items-center gap-1 rounded-md px-2.5 text-xs font-medium text-gray-600 hover:bg-gray-100"
        >
          <X className="h-3.5 w-3.5" />
          Temizle
        </button>
      )}
    </div>
  );
}

function clamp(v: number, min: number, max: number): number {
  if (Number.isNaN(v)) return min;
  return Math.max(min, Math.min(max, v));
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-gray-500">
        {label}
      </label>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={cn(
            "h-9 cursor-pointer appearance-none rounded-md border border-gray-300 bg-white pl-3 pr-8 text-sm text-gray-700",
            "hover:border-gray-400 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary-500/20",
          )}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <ChevronDown
          aria-hidden
          className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400"
        />
      </div>
    </div>
  );
}
