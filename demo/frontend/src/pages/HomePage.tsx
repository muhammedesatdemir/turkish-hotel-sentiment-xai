import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle, CheckCircle2, Eraser, Sparkles, Wand2 } from "lucide-react";
import toast from "react-hot-toast";

import { api } from "@/api/client";
import type { ExampleItem } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Spinner from "@/components/ui/Spinner";
import { Textarea } from "@/components/ui/Input";
import ExampleSelector from "@/components/examples/ExampleSelector";
import SentimentBadge from "@/components/prediction/SentimentBadge";
import ConfidenceMeter from "@/components/prediction/ConfidenceMeter";
import ProbabilityBars from "@/components/prediction/ProbabilityBars";
import LIMEHighlighter from "@/components/prediction/LIMEHighlighter";
import TopFeaturesTable from "@/components/prediction/TopFeaturesTable";
import { usePredict } from "@/hooks/usePredict";
import { useExplain } from "@/hooks/useExplain";

const MAX_WORDS = 1000;

export default function HomePage() {
  const [text, setText] = useState("");

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });
  const predict = usePredict();
  const explain = useExplain();

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;
  const tooLong = wordCount > MAX_WORDS;
  const canSubmit = wordCount > 0 && !tooLong && !predict.isPending;

  const handleAnalyze = () => {
    const trimmed = text.trim();
    if (!trimmed) {
      toast.error("Önce bir yorum girin");
      return;
    }
    if (tooLong) {
      toast.error(`En fazla ${MAX_WORDS} kelime`);
      return;
    }
    explain.reset();
    predict.mutate(trimmed);
  };

  const handleClear = () => {
    setText("");
    predict.reset();
    explain.reset();
  };

  const handleExample = (item: ExampleItem) => {
    setText(item.text);
    predict.reset();
    explain.reset();
  };

  const handleExplain = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    explain.mutate({ text: trimmed, num_samples: 300 });
  };

  return (
    <div className="space-y-6">
      <section className="rounded-xl bg-white p-6 shadow-sm sm:p-8">
        <h1 className="text-2xl font-bold text-gray-900 sm:text-3xl">
          Türkçe Otel Yorumu Duygu Analizi
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-gray-600 sm:text-base">
          Fine-tuned BERTurk modeli Türkçe otel yorumlarını{" "}
          <span className="font-semibold text-sentiment-negatif">negatif</span>,{" "}
          <span className="font-semibold text-sentiment-notr">nötr</span> veya{" "}
          <span className="font-semibold text-sentiment-pozitif">pozitif</span>{" "}
          olarak sınıflandırır. LIME ile hangi kelimelerin tahmini yönlendirdiğini görün.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge tone="primary">Macro-F1 0.880</Badge>
          <Badge tone="primary">F1-nötr 0.806</Badge>
          <Badge tone="primary">Accuracy %88.6</Badge>
          {health.data && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700">
              <CheckCircle2 className="h-3.5 w-3.5" />
              backend hazır · {health.data.device}
            </span>
          )}
          {health.isError && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700">
              <AlertCircle className="h-3.5 w-3.5" />
              backend bağlanılamadı (localhost:8000)
            </span>
          )}
        </div>
      </section>

      <Card>
        <CardHeader
          title="Yorum girişi"
          description="Otel yorumunuzu yazın veya bir örnek seçin"
          action={<ExampleSelector onSelect={handleExample} disabled={predict.isPending} />}
        />
        <CardBody className="space-y-3">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Otel yorumunuzu buraya yazın..."
            rows={5}
            disabled={predict.isPending}
            aria-label="Otel yorumu"
          />
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex gap-3 text-gray-500">
              <span>
                <b>{wordCount}</b> kelime
                {tooLong && <span className="ml-1 text-sentiment-negatif">(max {MAX_WORDS})</span>}
              </span>
              <span>
                <b>{charCount}</b> karakter
              </span>
            </div>
            <div className="flex gap-2">
              <Button variant="secondary" size="md" onClick={handleClear} disabled={!text || predict.isPending}>
                <Eraser className="h-4 w-4" />
                Temizle
              </Button>
              <Button size="md" onClick={handleAnalyze} loading={predict.isPending} disabled={!canSubmit}>
                <Sparkles className="h-4 w-4" />
                Analiz Et
              </Button>
            </div>
          </div>
        </CardBody>
      </Card>

      {predict.data && (
        <Card>
          <CardHeader
            title="Tahmin sonucu"
            description={`${predict.data.wc} kelime · ${predict.data.inference_time_ms} ms`}
          />
          <CardBody>
            <div className="grid gap-6 md:grid-cols-[auto,auto,1fr] md:items-center">
              <div className="flex justify-center">
                <SentimentBadge sentiment={predict.data.sentiment} size="lg" />
              </div>
              <div className="flex justify-center">
                <ConfidenceMeter value={predict.data.confidence} sentiment={predict.data.sentiment} />
              </div>
              <div className="min-w-0">
                <ProbabilityBars
                  probabilities={predict.data.probabilities}
                  predicted={predict.data.sentiment}
                />
              </div>
            </div>

            <div className="mt-6 flex items-center justify-between border-t border-gray-100 pt-4">
              <p className="text-sm text-gray-500">
                Modelin neden bu tahmini verdiğini görmek ister misiniz?
              </p>
              <Button
                variant="secondary"
                size="md"
                onClick={handleExplain}
                loading={explain.isPending}
                disabled={explain.isPending}
              >
                <Wand2 className="h-4 w-4" />
                {explain.data ? "XAI tekrar üret" : "XAI ile açıkla"}
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {predict.data && explain.isPending && (
        <Card>
          <CardBody className="space-y-3">
            <Spinner label="LIME açıklaması üretiliyor — model 300 perturbation üretiyor" />
            <p className="text-xs text-gray-500">
              Bu işlem GPU'da ~1-2 saniye, CPU'da ~6-10 saniye sürer.
            </p>
          </CardBody>
        </Card>
      )}

      {predict.data && explain.data && (
        <Card>
          <CardHeader
            title="LIME açıklaması"
            description={`${explain.data.num_samples} perturbation · ${explain.data.explain_time_ms} ms`}
          />
          <CardBody className="space-y-5">
            <LIMEHighlighter
              text={text.trim()}
              features={explain.data.lime_features[explain.data.sentiment] ?? []}
              predicted={explain.data.sentiment}
            />
            <div>
              <h4 className="mb-2 text-sm font-semibold text-gray-700">
                En etkili kelimeler (tahmin edilen sınıf için)
              </h4>
              <TopFeaturesTable
                featuresByClass={explain.data.lime_features}
                predicted={explain.data.sentiment}
                topK={6}
              />
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
