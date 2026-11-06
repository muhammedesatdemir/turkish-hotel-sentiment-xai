import { useMutation } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { ApiError, api } from "@/api/client";
import type { PredictResponse } from "@/api/types";

export function usePredict() {
  return useMutation<PredictResponse, ApiError, string>({
    mutationFn: (text: string) => api.predict(text),
    onError: (err) => {
      toast.error(err.message || "Tahmin başarısız");
    },
  });
}
