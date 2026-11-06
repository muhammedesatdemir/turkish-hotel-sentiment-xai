import { Link } from "react-router-dom";
import { ArrowLeft, Compass } from "lucide-react";
import Button from "@/components/ui/Button";

export default function NotFoundPage() {
  return (
    <div className="grid place-items-center py-20 text-center">
      <div className="max-w-md space-y-4">
        <div className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-primary-50 text-primary-700">
          <Compass className="h-8 w-8" />
        </div>
        <p className="font-mono text-6xl font-bold text-gray-300">404</p>
        <h1 className="text-xl font-semibold text-gray-900">Sayfa bulunamadı</h1>
        <p className="text-sm text-gray-500">
          Aradığınız sayfa mevcut değil veya taşınmış olabilir. Üst menüden bir
          bölüme geçebilir veya ana sayfaya dönebilirsiniz.
        </p>
        <div className="pt-2">
          <Link to="/">
            <Button variant="primary">
              <ArrowLeft className="h-4 w-4" />
              Ana sayfaya dön
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
