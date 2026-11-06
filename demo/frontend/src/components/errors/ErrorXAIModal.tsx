import { useEffect } from "react";
import { ArrowRight, Sparkles } from "lucide-react";

import type { ErrorRow } from "@/api/types";
import Modal from "@/components/ui/Modal";
import Spinner from "@/components/ui/Spinner";
import Button from "@/components/ui/Button";
import SentimentBadge from "@/components/prediction/SentimentBadge";
import ProbabilityBars from "@/components/prediction/ProbabilityBars";
import LIMEHighlighter from "@/components/prediction/LIMEHighlighter";
import TopFeaturesTable from "@/components/prediction/TopFeaturesTable";
import { useExplain } from "@/hooks/useExplain";

export default function ErrorXAIModal({
  row,
  onClose,
}: {
  row: ErrorRow | null;
  onClose: () => void;
}) {
  const explain = useExplain();
  const open = row !== null;

  useEffect(() => {
    if (row) {
      explain.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [row?.id]);

  const handleRun = () => {
    if (!row) return;
    explain.mutate({ text: row.text, num_samples: 300 });
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      size="xl"
      title={
        row && (
          <div className="flex flex-wrap items-center gap-2">
            <span>Hata #{row.id}</span>
            <SentimentBadge sentiment={row.true} size="sm" />
            <ArrowRight className="h-4 w-4 text-gray-400" />
            <SentimentBadge sentiment={row.pred} size="sm" />
          </div>
        )
      }
      description={row && `${row.wc} kelime · güven ${row.max_prob.toFixed(3)} · ${row.category}`}
    >
      {row && (
        <div className="space-y-5">
          <div>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
              Yorum
            </h4>
            <div className="rounded-md border border-gray-200 bg-gray-50 p-3 text-sm leading-relaxed text-gray-800">
              {row.text}
            </div>
          </div>

          <div>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
              Modelin verdiği olasılıklar
            </h4>
            <ProbabilityBars probabilities={row.probabilities} predicted={row.pred} />
          </div>

          {!explain.data && !explain.isPending && (
            <div className="flex items-center justify-between rounded-md border border-dashed border-gray-300 bg-gray-50 p-4">
              <p className="text-sm text-gray-600">
                Modelin neden <b>{row.true}</b> yerine <b>{row.pred}</b> dediğini kelime düzeyinde gör.
              </p>
              <Button onClick={handleRun} loading={explain.isPending}>
                <Sparkles className="h-4 w-4" />
                LIME ile aç
              </Button>
            </div>
          )}

          {explain.isPending && (
            <div className="flex items-center justify-center rounded-md border border-gray-200 bg-gray-50 p-6">
              <Spinner label="LIME hesaplanıyor (300 perturbation)" />
            </div>
          )}

          {explain.data && (
            <div className="space-y-4">
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                  LIME açıklaması ({explain.data.num_samples} sample · {explain.data.explain_time_ms} ms)
                </h4>
                <LIMEHighlighter
                  text={row.text}
                  features={explain.data.lime_features[explain.data.sentiment] ?? []}
                  predicted={explain.data.sentiment}
                />
              </div>

              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                  En etkili kelimeler — tahmin sınıfı için
                </h4>
                <TopFeaturesTable
                  featuresByClass={explain.data.lime_features}
                  predicted={explain.data.sentiment}
                  topK={6}
                />
              </div>

              {row.true !== explain.data.sentiment && explain.data.lime_features[row.true] && (
                <div>
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                    Karşılaştırma — gerçek sınıf <b>{row.true}</b> için en etkili kelimeler
                  </h4>
                  <TopFeaturesTable
                    featuresByClass={explain.data.lime_features}
                    predicted={row.true}
                    topK={6}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
