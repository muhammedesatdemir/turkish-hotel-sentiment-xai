import { useQueries } from "@tanstack/react-query";
import { AlertCircle, Brain, Target, Telescope, Users } from "lucide-react";

import { api } from "@/api/client";
import type {
  ClassicalBaseline,
  ConfusionMatrixData,
  CvResults,
  DatasetDistribution as DatasetData,
  EdaSummary,
  LeakageAudit,
  McnemarSummary,
  MultiModelData,
  OverviewMetrics,
  PerClassMetrics,
} from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";

import MetricCard from "@/components/dashboard/MetricCard";
import ConfusionMatrix from "@/components/dashboard/ConfusionMatrix";
import PerClassChart from "@/components/dashboard/PerClassChart";
import MultiModelTable from "@/components/dashboard/MultiModelTable";
import DatasetDistribution from "@/components/dashboard/DatasetDistribution";
import {
  CvValidationCard,
  McnemarCard,
  LeakageCard,
  ClassicalBaselineCard,
  EdaImagesCard,
} from "@/components/dashboard/StatisticalValidation";

function fmtPctMS(mean: number, std?: number | null) {
  const m = `${(mean * 100).toFixed(2)}%`;
  if (std === null || std === undefined) return m;
  return `${m} ± ${(std * 100).toFixed(2)}`;
}
function fmtNumMS(mean: number, std?: number | null) {
  const m = mean.toFixed(4);
  if (std === null || std === undefined) return m;
  return `${m} ± ${std.toFixed(4)}`;
}

export default function DashboardPage() {
  const queries = useQueries({
    queries: [
      { queryKey: ["overview"],        queryFn: api.overview,        staleTime: Infinity },
      { queryKey: ["confusion-matrix"], queryFn: api.confusionMatrix, staleTime: Infinity },
      { queryKey: ["per-class"],       queryFn: api.perClass,        staleTime: Infinity },
      { queryKey: ["multi-model"],     queryFn: api.multiModel,      staleTime: Infinity },
      { queryKey: ["dataset"],         queryFn: api.dataset,         staleTime: Infinity },
      { queryKey: ["cv"],               queryFn: api.cv,               staleTime: Infinity },
      { queryKey: ["mcnemar"],          queryFn: api.mcnemar,          staleTime: Infinity },
      { queryKey: ["classical"],        queryFn: api.classicalBaseline, staleTime: Infinity },
      { queryKey: ["leakage"],          queryFn: api.leakage,          staleTime: Infinity },
      { queryKey: ["eda"],              queryFn: api.eda,              staleTime: Infinity },
    ],
  });
  const [
    overview,
    cm,
    perClass,
    multi,
    dataset,
    cv,
    mcnemar,
    classical,
    leakage,
    eda,
  ] = queries;

  const anyError = queries.some((q) => q.isError);
  const allLoaded = queries.every((q) => q.data);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Dashboard</h1>
          <p className="mt-1 text-sm text-gray-600">
            BERTurk fine-tuned model — 3 seedlik eğitim sonuçları, multi-model karşılaştırma ve ablation
          </p>
        </div>
      </header>

      {anyError && (
        <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          <span>Bazı istatistikler yüklenemedi. Backend'in çalıştığından emin olun.</span>
        </div>
      )}

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-500">
          Genel performans
        </h2>
        {overview.data ? (
          <MetricCards data={overview.data as OverviewMetrics} />
        ) : (
          <SkeletonRow count={4} />
        )}
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Sınıf bazında metrikler"
            description="3-seed birleşik confusion matrix'ten yeniden hesaplandı (n=4180)"
          />
          <CardBody>
            {perClass.data ? (
              <PerClassChart data={perClass.data as PerClassMetrics} />
            ) : (
              <Spinner label="Yükleniyor…" />
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Confusion matrix"
            description="3-seed birleşik (n=4180), satır-bazlı normalize"
          />
          <CardBody>
            {cm.data ? (
              <ConfusionMatrix data={cm.data as ConfusionMatrixData} />
            ) : (
              <Spinner label="Yükleniyor…" />
            )}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="Multi-model karşılaştırması"
          description="4 ön-eğitilmiş modelin v3_genisletilmis veri seti üzerindeki 3-seed sonuçları (Macro-F1'e göre sıralı)"
        />
        <CardBody>
          {multi.data ? (
            <MultiModelTable data={multi.data as MultiModelData} />
          ) : (
            <Spinner label="Yükleniyor…" />
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Veri seti dağılımı"
          description="14.337 satır · sınıf, kaynak, train/val/test bölünmesi"
        />
        <CardBody>
          {dataset.data ? (
            <DatasetDistribution data={dataset.data as DatasetData} />
          ) : (
            <Spinner label="Yükleniyor…" />
          )}
        </CardBody>
      </Card>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-500">
          İstatistiksel Doğrulama
        </h2>
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="5-Fold Cross-Validation — BERTurk vs ELECTRA-tr"
              description="30 koşu (5 fold × 3 seed × 2 model) · veri varyansı altında stabilite doğrulaması"
            />
            <CardBody>
              {cv.data ? (
                <CvValidationCard data={cv.data as CvResults} />
              ) : (
                <Spinner label="Yükleniyor…" />
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title="McNemar İstatistiksel Anlamlılık Testi"
              description="33 bağımsız test (18 multi-model + 15 CV-fold) · BERT-ELECTRA çiftine yönelik 18 testin hiçbiri Bonferroni anlamlı değil"
            />
            <CardBody>
              {mcnemar.data ? (
                <McnemarCard data={mcnemar.data as McnemarSummary} />
              ) : (
                <Spinner label="Yükleniyor…" />
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title="Bağımsız Sızıntı Audit"
              description="3-yöntemli (exact + normalized + Jaccard) audit · 18 bağımsız split (3 multi-seed + 15 CV-fold)"
            />
            <CardBody>
              {leakage.data ? (
                <LeakageCard data={leakage.data as LeakageAudit} />
              ) : (
                <Spinner label="Yükleniyor…" />
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title="Klasik ML Baseline (TF-IDF + LogReg / LinearSVC)"
              description="9 koşu (3 seed × 3 model) · BERTurk +9,5 puan üstün · top features XAI bulgusunu doğruluyor"
            />
            <CardBody>
              {classical.data ? (
                <ClassicalBaselineCard data={classical.data as ClassicalBaseline} />
              ) : (
                <Spinner label="Yükleniyor…" />
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title="EDA — Veri Keşfi ve Pearson Korelasyon"
              description="10 meta-özellik · 5 görsel · sınıf ve kaynak dağılımı · görsele tıkla yeni sekmede tam boyut aç"
            />
            <CardBody>
              {eda.data ? (
                <EdaImagesCard data={eda.data as EdaSummary} />
              ) : (
                <Spinner label="Yükleniyor…" />
              )}
            </CardBody>
          </Card>
        </div>
      </section>

      {allLoaded && overview.data && (
        <p className="text-center text-xs text-gray-500">
          Model: <code className="rounded bg-gray-100 px-1 font-mono">{(overview.data as OverviewMetrics).model}</code>{" "}
          · Seçilen seed: {(overview.data as OverviewMetrics).chosen_seed} · Veri seti:{" "}
          {(overview.data as OverviewMetrics).dataset_size.toLocaleString("tr-TR")} yorum
        </p>
      )}
    </div>
  );
}

function MetricCards({ data }: { data: OverviewMetrics }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard
        accent="primary"
        icon={<Target className="h-4 w-4" />}
        label="Test Accuracy"
        value={fmtPctMS(data.test_accuracy.mean, data.test_accuracy.std)}
        sublabel="3-seed ortalama"
      />
      <MetricCard
        accent="primary"
        icon={<Brain className="h-4 w-4" />}
        label="Macro-F1"
        value={fmtNumMS(data.macro_f1.mean, data.macro_f1.std)}
        sublabel="3-seed ortalama"
      />
      <MetricCard
        accent="notr"
        icon={<Telescope className="h-4 w-4" />}
        label="F1-nötr"
        value={fmtNumMS(data.f1_notr.mean, data.f1_notr.std)}
        sublabel="zorlu sınıf · +0.18 sıçrama"
      />
      <MetricCard
        accent="pozitif"
        icon={<Users className="h-4 w-4" />}
        label="Test Boyutu"
        value={data.test_size_per_seed.toLocaleString("tr-TR")}
        sublabel={`3-seed birleşik ${data.test_size_combined_3seed.toLocaleString("tr-TR")}`}
      />
    </div>
  );
}

function SkeletonRow({ count }: { count: number }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="h-24 animate-pulse rounded-lg border border-gray-200 bg-white shadow-sm"
        />
      ))}
    </div>
  );
}
