import type { Sentiment } from "@/api/types";
import { SENTIMENT_LABEL } from "@/lib/constants";
import { cn } from "@/lib/utils";

const TONE: Record<Sentiment, string> = {
  pozitif: "bg-sentiment-pozitif text-white",
  "nötr":  "bg-sentiment-notr text-white",
  negatif: "bg-sentiment-negatif text-white",
};

export default function SentimentBadge({
  sentiment,
  size = "lg",
}: {
  sentiment: Sentiment;
  size?: "sm" | "md" | "lg";
}) {
  const SIZES = {
    sm: "px-2.5 py-0.5 text-xs",
    md: "px-3.5 py-1 text-sm",
    lg: "px-5 py-2 text-lg",
  } as const;
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center rounded-full font-bold tracking-wide",
        TONE[sentiment],
        SIZES[size],
      )}
      aria-label={`Duygu: ${sentiment}`}
    >
      {SENTIMENT_LABEL[sentiment]}
    </span>
  );
}
