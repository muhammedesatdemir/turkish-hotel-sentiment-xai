import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { api } from "@/api/client";
import type { ExampleItem } from "@/api/types";

export default function ExampleSelector({
  onSelect,
  disabled,
}: {
  onSelect: (item: ExampleItem) => void;
  disabled?: boolean;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["examples"],
    queryFn: api.examples,
    staleTime: Infinity,
  });

  return (
    <div className="relative inline-block">
      <select
        disabled={disabled || isLoading}
        defaultValue=""
        onChange={(e) => {
          const id = Number(e.target.value);
          const it = data?.items.find((x) => x.id === id);
          if (it) onSelect(it);
          e.target.value = "";
        }}
        className="h-10 cursor-pointer appearance-none rounded-md border border-gray-300 bg-white pl-3 pr-9 text-sm text-gray-700 hover:border-gray-400 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary-500/20 disabled:cursor-not-allowed disabled:bg-gray-50"
      >
        <option value="" disabled>
          {isLoading ? "Örnekler yükleniyor…" : "Hazır örnek seç…"}
        </option>
        {data?.items.map((it) => (
          <option key={it.id} value={it.id}>
            {it.label} · {it.wc} kelime
          </option>
        ))}
      </select>
      <ChevronDown
        aria-hidden
        className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400"
      />
    </div>
  );
}
