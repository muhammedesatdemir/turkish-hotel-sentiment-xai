import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import Layout from "@/components/layout/Layout";
import HomePage from "@/pages/HomePage";
import PageLoader from "@/components/layout/PageLoader";

const DashboardPage      = lazy(() => import("@/pages/DashboardPage"));
const BatchPage          = lazy(() => import("@/pages/BatchPage"));
const ErrorAnalysisPage  = lazy(() => import("@/pages/ErrorAnalysisPage"));
const AboutPage          = lazy(() => import("@/pages/AboutPage"));
const NotFoundPage       = lazy(() => import("@/pages/NotFoundPage"));

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route
          path="/dashboard"
          element={
            <Suspense fallback={<PageLoader />}>
              <DashboardPage />
            </Suspense>
          }
        />
        <Route
          path="/batch"
          element={
            <Suspense fallback={<PageLoader />}>
              <BatchPage />
            </Suspense>
          }
        />
        <Route
          path="/errors"
          element={
            <Suspense fallback={<PageLoader />}>
              <ErrorAnalysisPage />
            </Suspense>
          }
        />
        <Route
          path="/about"
          element={
            <Suspense fallback={<PageLoader />}>
              <AboutPage />
            </Suspense>
          }
        />
        <Route
          path="*"
          element={
            <Suspense fallback={<PageLoader />}>
              <NotFoundPage />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  );
}
