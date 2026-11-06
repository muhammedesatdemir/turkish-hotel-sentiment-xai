import { useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { AlertCircle } from "lucide-react";

import { api } from "@/api/client";
import type { ErrorRow } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";

import ErrorSummaryCards from "@/components/errors/ErrorSummaryCards";
import ErrorFilters, { DEFAULT_FILTERS, type FilterState } from "@/components/errors/ErrorFilters";
import ErrorTable from "@/components/errors/ErrorTable";
import ErrorXAIModal from "@/components/errors/ErrorXAIModal";

const PAGE_SIZE = 10;

export default function ErrorAnalysisPage() {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [offset, setOffset] = useState(0);
  const [activeRow, setActiveRow] = useState<ErrorRow | null>(null);

  const summary = useQuery({
    queryKey: ["error-summary"],
    queryFn: api.errorSummary,
    staleTime: Infinity,
  });

  const errorsQuery = useQuery({
    queryKey: ["errors", filters, offset],
    queryFn: () =>
      api.errors({
        direction: filters.direction !== "all" ? filters.direction : undefined,
        category:  filters.category  !== "all" ? filters.category  : undefined,
        min_conf:  filters.minConf > 0 ? filters.minConf : undefined,
        max_conf:  filters.maxConf < 1 ? filters.maxConf : undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    placeholderData: keepPreviousData,
  });

  const handleFilters = (next: FilterState) => {
    setFilters(next);
    setOffset(0);
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-gray-900">Hata Analizi</h1>
        <p className="mt-1 text-sm text-gray-600">
          Test setindeki 148 hatanın yön, kategori ve confidence dağılımı — herhangi bir kayda
          tıklayarak modelin kararını LIME ile açıklayın.
        </p>
      </header>

      {summary.isError && (
        <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          <span>Hata özeti yüklenemedi. Backend çalışıyor mu?</span>
        </div>
      )}

      {summary.data && <ErrorSummaryCards data={summary.data} />}

      <Card>
        <CardHeader title="Filtreler" />
        <CardBody>
          <ErrorFilters value={filters} onChange={handleFilters} />
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Hata kayıtları"
          description={
            errorsQuery.data
              ? `${errorsQuery.data.total} kayıt eşleşti · sayfa boyutu ${PAGE_SIZE}`
              : "Yükleniyor…"
          }
        />
        <CardBody>
          {errorsQuery.isLoading && !errorsQuery.data ? (
            <div className="flex justify-center py-8">
              <Spinner label="Yükleniyor…" />
            </div>
          ) : errorsQuery.isError ? (
            <p className="text-sm text-sentiment-negatif">Liste yüklenemedi.</p>
          ) : errorsQuery.data ? (
            <ErrorTable
              rows={errorsQuery.data.rows}
              total={errorsQuery.data.total}
              limit={errorsQuery.data.limit}
              offset={errorsQuery.data.offset}
              onPage={setOffset}
              onOpen={setActiveRow}
            />
          ) : null}
        </CardBody>
      </Card>

      <ErrorXAIModal row={activeRow} onClose={() => setActiveRow(null)} />
    </div>
  );
}
