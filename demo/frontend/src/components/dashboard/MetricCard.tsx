import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export default function MetricCard({
  label,
  value,
  sublabel,
  icon,
  accent = "primary",
}: {
  label: string;
  value: ReactNode;
  sublabel?: ReactNode;
  icon?: ReactNode;
  accent?: "primary" | "pozitif" | "notr" | "negatif";
}) {
  const ACCENT = {
    primary: "text-primary-700 bg-primary-50",
    pozitif: "text-emerald-700 bg-emerald-50",
    notr:    "text-gray-700 bg-gray-100",
    negatif: "text-red-700 bg-red-50",
  } as const;

  return (
    <div className="flex items-start gap-3 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      {icon && (
        <div className={cn("grid h-9 w-9 place-items-center rounded-md", ACCENT[accent])}>
          {icon}
        </div>
      )}
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium uppercase tracking-wide text-gray-500">{label}</p>
        <p className="mt-1 truncate font-mono text-2xl font-bold text-gray-900">{value}</p>
        {sublabel && <p className="mt-0.5 text-xs text-gray-500">{sublabel}</p>}
      </div>
    </div>
  );
}
