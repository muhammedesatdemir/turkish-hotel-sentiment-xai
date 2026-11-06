import { cn } from "@/lib/utils";

export default function Spinner({
  className,
  size = "md",
  label,
}: {
  className?: string;
  size?: "sm" | "md" | "lg";
  label?: string;
}) {
  const SIZE = { sm: "h-4 w-4", md: "h-6 w-6", lg: "h-8 w-8" }[size];
  return (
    <div className={cn("inline-flex items-center gap-2 text-gray-500", className)} role="status">
      <span
        aria-hidden
        className={cn(SIZE, "animate-spin rounded-full border-2 border-current border-t-transparent")}
      />
      {label && <span className="text-sm">{label}</span>}
    </div>
  );
}
