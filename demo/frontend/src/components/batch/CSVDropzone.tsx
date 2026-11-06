import { useCallback } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { File as FileIcon, UploadCloud, X } from "lucide-react";
import toast from "react-hot-toast";
import { cn } from "@/lib/utils";

const MAX_BYTES = 1_000_000; // 1 MB

export default function CSVDropzone({
  file,
  onFile,
  onClear,
  disabled,
}: {
  file: File | null;
  onFile: (f: File) => void;
  onClear: () => void;
  disabled?: boolean;
}) {
  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      if (rejected.length > 0) {
        toast.error(rejected[0].errors[0]?.message ?? "Dosya reddedildi");
        return;
      }
      const f = accepted[0];
      if (!f) return;
      if (f.size > MAX_BYTES) {
        toast.error(`Dosya 1 MB'ı aşmamalı (yüklenen: ${(f.size / 1024).toFixed(0)} KB)`);
        return;
      }
      onFile(f);
    },
    [onFile],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    disabled,
    accept: {
      "text/csv":   [".csv"],
      "text/plain": [".txt"],
    },
  });

  if (file) {
    return (
      <div className="flex items-center justify-between rounded-lg border border-gray-200 bg-white px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-md bg-primary-50 text-primary-700">
            <FileIcon className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-gray-900">{file.name}</p>
            <p className="text-xs text-gray-500">
              {(file.size / 1024).toFixed(1)} KB · {file.type || "text"}
            </p>
          </div>
        </div>
        <button
          onClick={onClear}
          disabled={disabled}
          className="rounded-md p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700 disabled:opacity-50"
          aria-label="Dosyayı kaldır"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed bg-white px-6 py-10 text-center transition-colors",
        isDragActive ? "border-primary bg-primary-50" : "border-gray-300 hover:border-primary-400",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      <input {...getInputProps()} />
      <UploadCloud className={cn("mb-3 h-10 w-10", isDragActive ? "text-primary" : "text-gray-400")} />
      <p className="text-sm font-medium text-gray-900">
        {isDragActive ? "Bırakabilirsiniz…" : "CSV veya TXT dosyası bırakın"}
      </p>
      <p className="mt-1 text-xs text-gray-500">
        veya tıklayıp seçin · max 50 satır · 1 MB
      </p>
    </div>
  );
}
