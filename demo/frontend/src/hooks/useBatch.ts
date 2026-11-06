import { useMutation } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { ApiError, api } from "@/api/client";
import type { BatchResponse } from "@/api/types";

export function useBatch() {
  return useMutation<BatchResponse, ApiError, string[]>({
    mutationFn: (texts) => api.batch(texts),
    onError: (err) => toast.error(err.message || "Toplu tahmin başarısız"),
  });
}

export function useBatchUpload() {
  return useMutation<BatchResponse, ApiError, File>({
    mutationFn: (file) => api.batchUpload(file),
    onError: (err) => toast.error(err.message || "CSV yüklemesi başarısız"),
  });
}
