import { useMemo, useState } from "react";
import { Download, FileSpreadsheet, Play, RotateCcw, Type } from "lucide-react";
import toast from "react-hot-toast";

import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { Textarea } from "@/components/ui/Input";
import CSVDropzone from "@/components/batch/CSVDropzone";
import BatchResultsTable from "@/components/batch/BatchResultsTable";
import BatchDistributionChart from "@/components/batch/BatchDistributionChart";
import { useBatch, useBatchUpload } from "@/hooks/useBatch";
import { batchToCsv, downloadCsv } from "@/lib/export";
import { cn } from "@/lib/utils";

const MAX_ROWS = 50;

type Mode = "upload" | "manual";

export default function BatchPage() {
  const [mode, setMode] = useState<Mode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [manualText, setManualText] = useState("");

  const upload = useBatchUpload();
  const batch = useBatch();
  const isPending = upload.isPending || batch.isPending;

  const result = mode === "upload" ? upload.data : batch.data;
  const error = mode === "upload" ? upload.error : batch.error;

  const manualLines = useMemo(
    () => manualText.split(/\r?\n/).map((s) => s.trim()).filter(Boolean),
    [manualText],
  );
  const manualOver = manualLines.length > MAX_ROWS;
  const canSubmitManual = manualLines.length > 0 && !manualOver && !isPending;
  const canSubmitUpload = !!file && !isPending;

  const handleSwitchMode = (m: Mode) => {
    if (isPending) return;
    setMode(m);
    upload.reset();
    batch.reset();
  };

  const handleRun = () => {
    if (mode === "upload") {
      if (!file) {
        toast.error("Önce bir dosya seçin");
        return;
      }
      batch.reset();
      upload.mutate(file);
    } else {
      if (manualLines.length === 0) {
        toast.error("En az 1 satır gir");
        return;
      }
      if (manualOver) {
        toast.error(`En fazla ${MAX_ROWS} satır`);
        return;
      }
      upload.reset();
      batch.mutate(manualLines);
    }
  };

  const handleReset = () => {
    setFile(null);
    setManualText("");
    upload.reset();
    batch.reset();
  };

  const handleExport = () => {
    if (!result || result.items.length === 0) return;
    const csv = batchToCsv(result.items);
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    downloadCsv(`tahmin_sonuclari_${stamp}.csv`, csv);
    toast.success("CSV indiriliyor");
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-gray-900">Toplu Analiz</h1>
        <p className="mt-1 text-sm text-gray-600">
          CSV/TXT yükleyin veya satır satır yapıştırın — model tüm yorumları sınıflandırsın (max {MAX_ROWS} satır).
        </p>
      </header>

      <Card>
        <CardHeader title="Giriş kaynağı" />
        <CardBody className="space-y-4">
          <div className="inline-flex rounded-md border border-gray-200 bg-white p-0.5">
            <ModeButton active={mode === "upload"} onClick={() => handleSwitchMode("upload")}>
              <FileSpreadsheet className="h-4 w-4" />
              CSV / TXT
            </ModeButton>
            <ModeButton active={mode === "manual"} onClick={() => handleSwitchMode("manual")}>
              <Type className="h-4 w-4" />
              Manuel giriş
            </ModeButton>
          </div>

          {mode === "upload" ? (
            <>
              <CSVDropzone
                file={file}
                onFile={setFile}
                onClear={() => {
                  setFile(null);
                  upload.reset();
                }}
                disabled={isPending}
              />
              <p className="text-xs text-gray-500">
                Format: tek sütunlu CSV (header opsiyonel: <code className="rounded bg-gray-100 px-1 font-mono">comment_text</code> /
                <code className="rounded bg-gray-100 px-1 font-mono"> text</code> /
                <code className="rounded bg-gray-100 px-1 font-mono"> yorum</code>) veya her satırda bir yorum.
              </p>
            </>
          ) : (
            <>
              <Textarea
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                placeholder={"Her satıra bir yorum yazın\n\nÖrnek:\nKonum harikaydı, personel ilgili.\nOda kirliydi.\nFiyat performansı orta."}
                rows={8}
                disabled={isPending}
              />
              <p className="text-xs text-gray-500">
                <b>{manualLines.length}</b> satır
                {manualOver && <span className="ml-1 text-sentiment-negatif">(max {MAX_ROWS})</span>}
              </p>
            </>
          )}

          <div className="flex flex-wrap items-center justify-between gap-2">
            <Button
              variant="secondary"
              size="md"
              onClick={handleReset}
              disabled={isPending || (!file && !manualText && !result)}
            >
              <RotateCcw className="h-4 w-4" />
              Sıfırla
            </Button>
            <Button
              size="md"
              onClick={handleRun}
              loading={isPending}
              disabled={mode === "upload" ? !canSubmitUpload : !canSubmitManual}
            >
              <Play className="h-4 w-4" />
              {isPending ? "İşleniyor…" : "Analizi Başlat"}
            </Button>
          </div>
        </CardBody>
      </Card>

      {error && (
        <Card>
          <CardBody>
            <p className="text-sm text-sentiment-negatif">{error.message}</p>
          </CardBody>
        </Card>
      )}

      {result && (
        <>
          <Card>
            <CardHeader
              title="Özet"
              description={`${result.total} yorum analiz edildi · ${result.inference_time_ms} ms`}
              action={
                <Button variant="secondary" size="sm" onClick={handleExport}>
                  <Download className="h-4 w-4" />
                  CSV indir
                </Button>
              }
            />
            <CardBody>
              <BatchDistributionChart distribution={result.distribution} items={result.items} />
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Detaylı sonuçlar" description="Sıralanabilir, filtrelenebilir, aranabilir" />
            <CardBody>
              <BatchResultsTable items={result.items} />
            </CardBody>
          </Card>
        </>
      )}
    </div>
  );
}

function ModeButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm px-3 py-1.5 text-sm font-medium transition-colors",
        active ? "bg-primary text-white" : "text-gray-600 hover:bg-gray-100",
      )}
    >
      {children}
    </button>
  );
}
