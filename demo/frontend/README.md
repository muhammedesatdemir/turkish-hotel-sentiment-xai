# Frontend — Vite + React + TypeScript

## Çalıştır

```powershell
cd demo/frontend
npm install
npm run dev
```

<http://localhost:5173/> üzerinde açılır. Backend (`localhost:8000`) çalışıyor olmalı.

## Diğer komutlar

```powershell
npm run typecheck   # tsc --noEmit
npm run build       # tsc -b && vite build
npm run preview     # built dist/'i lokalde sunar
```

## Yapı

```
src/
├── api/
│   ├── client.ts      # 14 endpoint wrapper (fetch + ApiError)
│   └── types.ts       # Backend response shape'leri
├── components/
│   ├── ui/            # Button, Card, Badge, Spinner, Input, Modal
│   ├── layout/        # Layout, Navbar, Footer, PageLoader
│   ├── prediction/    # SentimentBadge, ProbabilityBars, ConfidenceMeter, LIMEHighlighter, TopFeaturesTable
│   ├── examples/      # ExampleSelector
│   ├── dashboard/     # MetricCard, ConfusionMatrix, PerClassChart, MultiModelTable, AblationTable, DatasetDistribution
│   ├── batch/         # CSVDropzone, BatchResultsTable, BatchDistributionChart
│   └── errors/        # ErrorSummaryCards, ErrorFilters, ErrorTable, ErrorXAIModal
├── hooks/             # usePredict, useExplain, useBatch (TanStack mutations)
├── lib/               # utils.ts (cn, format), constants.ts (sentiment palette), export.ts (CSV)
├── pages/             # HomePage, DashboardPage, BatchPage, ErrorAnalysisPage, AboutPage, NotFoundPage
├── App.tsx            # Router (lazy routes + Suspense)
├── main.tsx           # QueryClient, BrowserRouter, Toaster
└── index.css          # Tailwind directives, Inter font
```

## Stack

- **Vite 5** — build tool
- **React 18** + **TypeScript 5** + **Tailwind CSS 3**
- **TanStack Query 5** — API state, caching
- **React Router 6** — SPA routing
- **Recharts** — pie/bar grafikleri (lazy chunk)
- **Lucide React** — ikon seti
- **react-hot-toast** — toast bildirimler
- **react-dropzone** — CSV upload

## Yapılandırma

- `vite.config.ts` — `@/` → `src/` alias; `manualChunks` ile vendor splitting (recharts ayrı chunk)
- `tailwind.config.ts` — `primary` (#1e40af), `sentiment-{pozitif,notr,negatif}` paleti, Inter / JetBrains Mono fontları
- API base URL: `VITE_API_BASE_URL` env var (varsayılan `http://localhost:8000`)

## Lazy loading

`HomePage` eager yüklenir; `Dashboard`, `Batch`, `Errors`, `About`, `404` lazy.
İlk yüklemede yalnızca tahmin akışı için gerekli kod indirilir; Recharts ve modal kodu sayfa ziyaret edilince çekilir.
