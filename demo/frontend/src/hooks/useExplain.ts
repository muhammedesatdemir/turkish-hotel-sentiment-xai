import { useMutation } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { ApiError, api } from "@/api/client";
import type { ExplainResponse } from "@/api/types";

export function useExplain() {
  return useMutation<ExplainResponse, ApiError, { text: string; num_samples?: number }>({
    mutationFn: ({ text, num_samples }) => api.explain(text, num_samples),
    onError: (err) => {
      toast.error(err.message || "LIME açıklaması üretilemedi");
    },
  });
}
