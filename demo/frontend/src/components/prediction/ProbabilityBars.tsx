import type { Sentiment } from "@/api/types";
import { LABELS, SENTIMENT_COLORS, SENTIMENT_LABEL } from "@/lib/constants";
import { formatPercent } from "@/lib/utils";

export default function ProbabilityBars({
  probabilities,
  predicted,
}: {
  probabilities: Record<string, number>;
  predicted: Sentiment;
}) {
  return (
    <div className="space-y-3">
      {LABELS.map((label) => {
        const value = probabilities[label] ?? 0;
        const isPred = label === predicted;
        return (
          <div key={label}>
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className={isPred ? "font-semibold text-gray-900" : "text-gray-600"}>
                {SENTIMENT_LABEL[label]}
              </span>
              <span className={"font-mono " + (isPred ? "font-semibold text-gray-900" : "text-gray-600")}>
                {formatPercent(value, 2)}
              </span>
            </div>
            <div className="h-3 w-full overflow-hidden rounded-full bg-gray-100">
              <div
                className="h-full rounded-full transition-all duration-500 ease-out"
                style={{
                  width: `${Math.max(2, value * 100)}%`,
                  backgroundColor: SENTIMENT_COLORS[label],
                  opacity: isPred ? 1 : 0.6,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
