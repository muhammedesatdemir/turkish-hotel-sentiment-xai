import { CheckCircle2, AlertCircle, Activity, ImageIcon } from "lucide-react";
import type {
  ClassicalBaseline,
  CvResults,
  EdaSummary,
  LeakageAudit,
  McnemarSummary,
  SarcasmAnalysis,
} from "@/api/types";

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

// =============================================================================
// 5-Fold Cross-Validation kartı
// =============================================================================
export function CvValidationCard({ data }: { data: CvResults }) {
  const bert = data.models.find((m) => m.key === "berturk");
  const electra = data.models.find((m) => m.key === "electra_tr");

  if (!bert || !electra) {
    return <p className="text-sm text-gray-500">CV verisi yüklenemedi.</p>;
  }

  return (
    <div className="space-y-3">
      <div className="overflow-hidden rounded-md border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Model</th>
              <th className="px-3 py-2 text-right font-medium">Macro-F1</th>
              <th className="px-3 py-2 text-right font-medium">F1-nötr</th>
              <th className="px-3 py-2 text-right font-medium">Accuracy</th>
              <th className="px-3 py-2 text-right font-medium">Süre</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {[bert, electra].map((m) => (
              <tr key={m.key} className={m.highlighted ? "bg-primary-50/40" : ""}>
                <td className="px-3 py-2 font-medium text-gray-900">
                  {m.name}
                  {m.highlighted && (
                    <span className="ml-2 rounded bg-primary-100 px-1.5 py-0.5 text-[10px] font-semibold text-primary-700">
                      SEÇİLEN
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right font-mono">
                  {m.metrics.macro_f1.mean.toFixed(4)} ±{" "}
                  {m.metrics.macro_f1.std.toFixed(4)}
                </td>
                <td className="px-3 py-2 text-right font-mono">
                  {m.metrics.f1_notr.mean.toFixed(4)} ±{" "}
                  {m.metrics.f1_notr.std.toFixed(4)}
                </td>
                <td className="px-3 py-2 text-right font-mono">
                  {m.metrics.accuracy.mean.toFixed(4)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-600">
                  {m.elapsed_hours.toFixed(2)}h
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-2 text-xs sm:grid-cols-3">
        <div className="rounded-md bg-gray-50 p-2.5">
          <p className="font-semibold text-gray-700">Δ Macro-F1</p>
          <p className="font-mono text-gray-900">
            +{data.comparison.delta_macro_f1.toFixed(4)}
          </p>
          <p className="text-gray-500">σ ile aynı düzey</p>
        </div>
        <div className="rounded-md bg-gray-50 p-2.5">
          <p className="font-semibold text-gray-700">Hız avantajı</p>
          <p className="font-mono text-gray-900">
            {data.comparison.speed_ratio.toFixed(1)}×
          </p>
          <p className="text-gray-500">BERTurk daha hızlı</p>
        </div>
        <div className="rounded-md bg-gray-50 p-2.5">
          <p className="font-semibold text-gray-700">Toplam koşu</p>
          <p className="font-mono text-gray-900">{data.n_runs_per_model * 2}</p>
          <p className="text-gray-500">5 fold × 3 seed × 2 model</p>
        </div>
      </div>

      <p className="rounded-md border-l-2 border-primary-300 bg-primary-50/40 p-2.5 text-xs italic text-gray-700">
        {data.comparison.interpretation}
      </p>
      <p className="text-xs text-gray-500">{data.vs_multi_seed.note}</p>
    </div>
  );
}

// =============================================================================
// McNemar İstatistiksel Anlamlılık kartı
// =============================================================================
export function McnemarCard({ data }: { data: McnemarSummary }) {
  const be = data.multi_model.bert_vs_electra;
  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            Toplam test
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            {data.total_tests}
          </p>
          <p className="text-xs text-gray-500">
            18 multi-model + 15 CV-fold
          </p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            BERT-ELECTRA testi
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            {data.bert_electra_tests}
          </p>
          <p className="text-xs text-gray-500">3 multi-seed + 15 CV-fold</p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            CV Fisher's combined p
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            {data.cv_fold.fishers_combined_p.toFixed(3)}
          </p>
          <p className="text-xs text-emerald-700">eşdeğerlik (p &gt; 0,05)</p>
        </div>
      </div>

      <div className="overflow-hidden rounded-md border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Çift</th>
              <th className="px-3 py-2 text-right font-medium">min p</th>
              <th className="px-3 py-2 text-center font-medium" title="Bonferroni m=3">
                Bonf m=3
              </th>
              <th className="px-3 py-2 text-center font-medium" title="Bonferroni m=18">
                Bonf m=18
              </th>
              <th className="px-3 py-2 text-left font-medium">Yorum</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {[
              { label: "BERTurk vs ELECTRA-tr", v: be },
              { label: "BERTurk vs mBERT", v: data.multi_model.bert_vs_mbert },
              { label: "BERTurk vs XLM-R", v: data.multi_model.bert_vs_xlmr },
            ].map((row) => (
              <tr key={row.label}>
                <td className="px-3 py-2 font-medium text-gray-800">{row.label}</td>
                <td className="px-3 py-2 text-right font-mono">
                  {row.v.min_p_value.toFixed(5)}
                </td>
                <td className="px-3 py-2 text-center">
                  <BonferroniBadge sig={row.v.significant_m3} />
                </td>
                <td className="px-3 py-2 text-center">
                  <BonferroniBadge sig={row.v.significant_m18} />
                </td>
                <td className="px-3 py-2 text-xs text-gray-600">
                  {row.v.interpretation}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="rounded-md border-l-2 border-primary-300 bg-primary-50/40 p-2.5 text-xs italic text-gray-700">
        {data.akademik_sonuc}
      </p>
    </div>
  );
}

function BonferroniBadge({ sig }: { sig: boolean }) {
  return sig ? (
    <span className="inline-flex items-center gap-1 rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700">
      <CheckCircle2 className="h-3 w-3" /> EVET
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-semibold text-gray-600">
      hayır
    </span>
  );
}

// =============================================================================
// Sızıntı Audit kartı
// =============================================================================
export function LeakageCard({ data }: { data: LeakageAudit }) {
  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            Bağımsız split
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">18</p>
          <p className="text-xs text-gray-500">3 multi-seed + 15 CV-fold</p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            Exact match invariant
          </p>
          <p className="mt-1 inline-flex items-center gap-1 font-mono text-2xl font-bold text-emerald-700">
            <CheckCircle2 className="h-5 w-5" /> 0
          </p>
          <p className="text-xs text-gray-500">tüm 18 split</p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            Max etki bound
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            ≤ {data.impact_bound.max_impact_macro_f1_multi_seed.toFixed(3)}
          </p>
          <p className="text-xs text-gray-500">σ ile aynı mertebe</p>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="overflow-hidden rounded-md border border-gray-200">
          <p className="border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-gray-600">
            Multi-seed (3 final eğitim)
          </p>
          <table className="w-full text-sm">
            <thead className="text-xs text-gray-500">
              <tr>
                <th className="px-3 py-1.5 text-left font-medium">Seed</th>
                <th className="px-3 py-1.5 text-right font-medium">Exact</th>
                <th className="px-3 py-1.5 text-right font-medium">Norm</th>
                <th className="px-3 py-1.5 text-right font-medium">J≥0,90</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {data.multi_seed.map((s) => (
                <tr key={s.seed}>
                  <td className="px-3 py-1.5 font-mono text-gray-700">{s.seed}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-emerald-700">
                    {s.exact_tt}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-gray-600">
                    {s.normalized_tt}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono">
                    {s.jaccard_090_tt}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="overflow-hidden rounded-md border border-gray-200">
          <p className="border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-gray-600">
            5-fold CV (15 fold) — agregat
          </p>
          <dl className="space-y-1.5 p-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-gray-600">Exact (15/15)</dt>
              <dd className="font-mono text-emerald-700">
                <CheckCircle2 className="inline h-3.5 w-3.5" /> 0
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-600">Normalized μ ± σ</dt>
              <dd className="font-mono text-gray-700">
                {data.cv_15_fold.normalized_mean.toFixed(1)} ±{" "}
                {data.cv_15_fold.normalized_std.toFixed(1)}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-600">Jaccard ≥ 0,90 μ ± σ</dt>
              <dd className="font-mono text-gray-900">
                {data.cv_15_fold.j090_mean.toFixed(1)} ±{" "}
                {data.cv_15_fold.j090_std.toFixed(1)}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-600">Jaccard ≥ 0,90 [min, max]</dt>
              <dd className="font-mono text-gray-700">
                [{data.cv_15_fold.j090_min}, {data.cv_15_fold.j090_max}]
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <p className="rounded-md border-l-2 border-primary-300 bg-primary-50/40 p-2.5 text-xs italic text-gray-700">
        {data.impact_bound.interpretation}
      </p>
    </div>
  );
}

// =============================================================================
// Klasik ML Baseline kartı
// =============================================================================
export function ClassicalBaselineCard({ data }: { data: ClassicalBaseline }) {
  const best = data.models.find((m) => m.best_classical);
  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">BERTurk</p>
          <p className="mt-1 font-mono text-2xl font-bold text-primary-700">
            {data.vs_berturk.berturk_macro_f1.toFixed(4)}
          </p>
          <p className="text-xs text-gray-500">Macro-F1 (multi-seed)</p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            En iyi klasik
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            {data.vs_berturk.best_classical_macro_f1.toFixed(4)}
          </p>
          <p className="text-xs text-gray-500 truncate">{best?.name}</p>
        </div>
        <div className="rounded-md border border-emerald-300 bg-emerald-50 p-3">
          <p className="text-xs uppercase tracking-wide text-emerald-700">
            Δ üstünlük
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-emerald-800">
            +{(data.vs_berturk.delta_macro_f1 * 100).toFixed(1)} puan
          </p>
          <p className="text-xs text-emerald-700">
            F1-nötr: +{(data.vs_berturk.delta_f1_notr * 100).toFixed(1)} puan
          </p>
        </div>
      </div>

      <div className="overflow-hidden rounded-md border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Klasik model</th>
              <th className="px-3 py-2 text-right font-medium">Macro-F1</th>
              <th className="px-3 py-2 text-right font-medium">F1-nötr</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {data.models.map((m) => (
              <tr key={m.key} className={m.best_classical ? "bg-amber-50/50" : ""}>
                <td className="px-3 py-1.5 text-gray-800">
                  {m.name}
                  {m.best_classical && (
                    <span className="ml-1 text-amber-600">★</span>
                  )}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {m.metrics.macro_f1.mean.toFixed(4)} ±{" "}
                  {m.metrics.macro_f1.std.toFixed(4)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {m.metrics.f1_notr.mean.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {(["negatif", "nötr", "pozitif"] as const).map((cls) => {
          const top = data.top_features_per_class[cls];
          if (!top) return null;
          return (
            <div
              key={cls}
              className="rounded-md border border-gray-200 bg-gray-50/50 p-2.5"
            >
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                {cls} top-5
              </p>
              <ul className="mt-1.5 space-y-0.5 text-xs">
                {top.map(([word, coef]) => (
                  <li key={word} className="flex justify-between">
                    <span className="font-mono text-gray-800">{word}</span>
                    <span className="font-mono text-gray-500">
                      {coef.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>

      <p className="rounded-md border-l-2 border-primary-300 bg-primary-50/40 p-2.5 text-xs italic text-gray-700">
        <Activity className="mr-1 inline h-3.5 w-3.5" />
        {data.xai_cross_validation}
      </p>
    </div>
  );
}

// =============================================================================
// Sarkazm/İroni Sınırlaması kartı
// =============================================================================
export function SarcasmLimitationCard({ data }: { data: SarcasmAnalysis }) {
  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-red-200 bg-red-50 p-3">
          <p className="text-xs uppercase tracking-wide text-red-700">
            Sarkastik yanlış
          </p>
          <p className="mt-1 inline-flex items-center gap-1 font-mono text-2xl font-bold text-red-800">
            <AlertCircle className="h-5 w-5" />
            {data.n_sarcastic_wrong}/{data.n_sarcastic}
          </p>
          <p className="text-xs text-red-700">yapısal sınır kanıtı</p>
        </div>
        <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3">
          <p className="text-xs uppercase tracking-wide text-emerald-700">
            Kontrol grubu
          </p>
          <p className="mt-1 inline-flex items-center gap-1 font-mono text-2xl font-bold text-emerald-800">
            <CheckCircle2 className="h-5 w-5" />
            {data.control_correct ? "DOĞRU" : "YANLIŞ"}
          </p>
          <p className="text-xs text-emerald-700">temel görev OK</p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            Yanlış güveni
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            {(data.avg_confidence_when_wrong * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500">model overconfident</p>
        </div>
      </div>

      <div className="overflow-hidden rounded-md border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase tracking-wide text-gray-500">
            <tr>
              <th className="px-3 py-1.5 text-left font-medium">#</th>
              <th className="px-3 py-1.5 text-left font-medium">Metin (kısalt)</th>
              <th className="px-3 py-1.5 text-center font-medium">Gerçek</th>
              <th className="px-3 py-1.5 text-center font-medium">Tahmin</th>
              <th className="px-3 py-1.5 text-right font-medium">Güven</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {data.examples.map((ex) => {
              const correct = ex.true === ex.pred;
              return (
                <tr key={ex.id} className={correct ? "" : "bg-red-50/30"}>
                  <td className="px-3 py-1.5 font-mono text-gray-500">{ex.id}</td>
                  <td className="px-3 py-1.5 text-xs text-gray-700">
                    {ex.text_short}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <SentimentBadge label={ex.true} />
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <SentimentBadge label={ex.pred} dim={!correct} />
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-xs">
                    {(ex.confidence * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="rounded-md border-l-2 border-primary-300 bg-primary-50/40 p-2.5 text-xs italic text-gray-700">
        {data.interpretation}
      </p>
    </div>
  );
}

function SentimentBadge({ label, dim = false }: { label: string; dim?: boolean }) {
  const colors: Record<string, string> = {
    negatif: "bg-red-100 text-red-700",
    "nötr": "bg-gray-100 text-gray-700",
    pozitif: "bg-emerald-100 text-emerald-700",
  };
  const cls = colors[label] ?? "bg-gray-100 text-gray-700";
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold ${cls} ${
        dim ? "opacity-60" : ""
      }`}
    >
      {label}
    </span>
  );
}

// =============================================================================
// EDA Görselleri kartı (Pearson korelasyon, sınıf dağılımı, kelime cloud vb.)
// =============================================================================
export function EdaImagesCard({ data }: { data: EdaSummary }) {
  const images = data.images ?? [];
  const pearsonRs =
    data.key_correlations?.["rating_auto_x_sentiment_id"]?.pearson_r;
  const isAiNotr = data.key_correlations?.["is_ai_x_is_notr"]?.point_biserial_r;

  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            rating ↔ sentiment
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            r = {pearsonRs?.toFixed(4) ?? "—"}
          </p>
          <p className="text-xs text-gray-500">etiketleme tutarlılığı</p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            is_ai ↔ is_nötr
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            r = {isAiNotr?.toFixed(3) ?? "—"}
          </p>
          <p className="text-xs text-gray-500">point-biserial (AI→nötr)</p>
        </div>
        <div className="rounded-md border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            Meta-özellik
          </p>
          <p className="mt-1 font-mono text-2xl font-bold text-gray-900">
            {data.n_meta_features}
          </p>
          <p className="text-xs text-gray-500">korelasyon matrisinde</p>
        </div>
      </div>

      {images.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2">
          {images.map((img) => (
            <figure
              key={img.url}
              className="rounded-md border border-gray-200 bg-white p-2.5"
            >
              <a
                href={`${API_BASE}${img.url}`}
                target="_blank"
                rel="noreferrer"
                className="block"
                title="Yeni sekmede tam boyut aç"
              >
                <img
                  src={`${API_BASE}${img.url}`}
                  alt={img.title}
                  className="h-auto w-full rounded border border-gray-100 transition hover:opacity-90"
                  loading="lazy"
                />
              </a>
              <figcaption className="mt-2 space-y-0.5">
                <p className="text-xs font-semibold text-gray-800">
                  <ImageIcon className="mr-1 inline h-3 w-3 text-gray-500" />
                  {img.title}
                </p>
                <p className="text-xs text-gray-600">{img.description}</p>
              </figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <p className="rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
          EDA görselleri bulunamadı. Backend'in <code>EDA_DIR</code> mount'u
          başarısız olabilir; <code>reports/eda/*.png</code> mevcut mu kontrol
          edin.
        </p>
      )}
    </div>
  );
}
