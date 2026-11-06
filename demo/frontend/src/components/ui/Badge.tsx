import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type Tone = "default" | "primary" | "pozitif" | "notr" | "negatif" | "warning";

interface Props extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

const TONE_CLASSES: Record<Tone, string> = {
  default: "bg-gray-100 text-gray-800",
  primary: "bg-primary-100 text-primary-800",
  pozitif: "bg-emerald-100 text-emerald-800",
  notr:    "bg-gray-200 text-gray-800",
  negatif: "bg-red-100 text-red-800",
  warning: "bg-amber-100 text-amber-800",
};

export default function Badge({ className, tone = "default", ...rest }: Props) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        TONE_CLASSES[tone],
        className,
      )}
      {...rest}
    />
  );
}
