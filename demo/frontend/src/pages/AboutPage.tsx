import {
  BookOpen,
  Brain,
  CheckCircle2,
  Database,
  GitBranch,
  GraduationCap,
  Lightbulb,
  ShieldCheck,
  Users,
} from "lucide-react";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";

export default function AboutPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-gray-900">Proje Hakkında</h1>
        <p className="mt-1 text-sm text-gray-600">
          Bitirme projesi · İstinye Üniversitesi · 2026
        </p>
      </header>

      <Card>
        <CardHeader title="Özet" />
        <CardBody className="space-y-3 text-sm leading-relaxed text-gray-700">
          <p>
            Türkçe otel yorumlarında <b>negatif / nötr / pozitif</b> üç-sınıflı duygu
            sınıflandırması yapan, fine-tuned <b>BERTurk</b> tabanlı bir sistemin web
            demosu. <b>14.337 satırlık</b> veri seti üzerinde 3-seed multi-seed eğitim ile{" "}
            <b>Macro-F1 = 0,8797 ± 0,0073</b> elde edildi; nötr sınıfının F1 değeri{" "}
            <b>0,8056</b> (önceki iterasyona göre +12 puan iyileşme).
          </p>
          <p>
            Sonuçlar bağımsız <b>5-fold cross-validation</b> (15 koşu, BERTurk + ELECTRA-tr
            için ayrı ayrı = 30 toplam koşu) ile doğrulandı; <b>33 McNemar testi</b>{" "}
            BERTurk-ELECTRA arasındaki farkın istatistiksel olarak ayırt edilemediğini
            gösterdi. Veri sızıntısı 18 bağımsız split (3 multi-seed + 15 CV-fold)
            boyunca <b>Exact = 0</b> invariant. Klasik TF-IDF + LogReg baseline'ı
            BERTurk'ün <b>+9,5 puan</b> üstünlüğünü kanıtladı.
          </p>
          <p className="text-xs text-gray-500">
            Demo; tek yorum tahmini + LIME XAI, multi-model + CV karşılaştırmaları,
            McNemar/sızıntı/baseline/sarkazm istatistikleri, toplu CSV analizi ve test
            setindeki 148 hatanın detaylı incelemesini sunar.
          </p>
        </CardBody>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader
            title={
              <span className="inline-flex items-center gap-2">
                <Brain className="h-4 w-4" /> Yöntem
              </span>
            }
          />
          <CardBody className="space-y-2 text-sm text-gray-700">
            <p>
              Pre-trained <b>dbmdz/bert-base-turkish-cased</b> modeli sequence-classification
              başlığıyla fine-tune edildi.
            </p>
            <ul className="ml-4 list-disc space-y-1 text-gray-600">
              <li>Max sequence length: 256 token</li>
              <li>5 epoch, learning rate 2e-5, weighted cross-entropy loss</li>
              <li>FP16 mixed precision, deterministic seeds (42, 43, 44)</li>
              <li>Train / Val / Test = 11.781 / 1.178 / 1.378 (per-class split)</li>
              <li>AI üretimi yorumlar yalnız train setinde, test seti %100 gerçek</li>
            </ul>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title={
              <span className="inline-flex items-center gap-2">
                <Database className="h-4 w-4" /> Veri seti
              </span>
            }
          />
          <CardBody className="space-y-2 text-sm text-gray-700">
            <p>
              <b>14.337</b> Türkçe otel yorumu — kaynak şeffaflığı:
            </p>
            <ul className="ml-4 list-disc space-y-1 text-gray-600">
              <li>
                <b>otelpuan_otelyorum</b> — 11.086 (%77,3) · doğal kullanıcı yorumu
              </li>
              <li>
                <b>ai_generated</b> — 2.500 (%17,4) · sadece train, test'e sızdırılmadı
              </li>
              <li>
                <b>manuel_toplama</b> — 751 (%5,2) · elle-yazımlı yapı-temelli (yapay zekâ
                üretimi DEĞİL)
              </li>
            </ul>
            <div className="mt-2 rounded-md bg-gray-50 p-2.5 text-xs">
              <p className="font-semibold text-gray-700">Sınıf dağılımı:</p>
              <p className="text-gray-600">
                pozitif <b>5.307</b> (%37,0) · nötr <b>4.660</b> (%32,5) · negatif{" "}
                <b>4.370</b> (%30,5)
              </p>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title={
              <span className="inline-flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" /> İstatistiksel doğrulama
              </span>
            }
          />
          <CardBody className="space-y-2 text-sm text-gray-700">
            <ul className="ml-4 list-disc space-y-1 text-gray-600">
              <li>
                <b>5-fold cross-validation</b> (BERTurk + ELECTRA-tr, 15+15 = 30 koşu)
              </li>
              <li>
                <b>33 McNemar testi</b> (18 multi-model + 15 CV-fold) — BERT-ELECTRA
                eşdeğer
              </li>
              <li>
                <b>3-yöntemli sızıntı audit</b> (exact + normalized + Jaccard) — 18
                bağımsız split
              </li>
              <li>
                <b>Klasik baseline</b> (TF-IDF + LogReg/SVC) — BERTurk +9,5 puan üstün
              </li>
              <li>
                <b>EDA + Pearson korelasyon</b> — rating ↔ sentiment r = +0,813 (etiket
                tutarlılığı)
              </li>
            </ul>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title={
              <span className="inline-flex items-center gap-2">
                <Lightbulb className="h-4 w-4" /> Açıklanabilirlik
              </span>
            }
          />
          <CardBody className="space-y-2 text-sm text-gray-700">
            <p>
              <b>LIME</b> (Local Interpretable Model-Agnostic Explanations) ile her tahmin
              için kelime düzeyinde katkı analizi.
            </p>
            <ul className="ml-4 list-disc space-y-1 text-gray-600">
              <li>BoW kapalı, pozisyonel açıklama (num_samples=300)</li>
              <li>Sınıf başına en etkili kelimeler tablosu</li>
              <li>148 hata için interaktif XAI modal</li>
              <li>
                <b>Sarkazm sınırlaması:</b> 8 sentetik örnek, 6/7 yanlış (yapısal sınır
                kanıtı)
              </li>
            </ul>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title={
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4" /> Anahtar sonuçlar
              </span>
            }
          />
          <CardBody className="text-sm text-gray-700">
            <dl className="grid gap-2 sm:grid-cols-2">
              <div>
                <dt className="text-xs uppercase tracking-wide text-gray-500">
                  Macro-F1 (multi-seed)
                </dt>
                <dd className="font-mono font-semibold text-gray-900">
                  0,8797 ± 0,0073
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-gray-500">
                  Macro-F1 (5-fold CV)
                </dt>
                <dd className="font-mono font-semibold text-gray-900">
                  0,8750 ± 0,0066
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-gray-500">
                  F1-nötr (multi-seed)
                </dt>
                <dd className="font-mono font-semibold text-gray-900">
                  0,8056 ± 0,0112
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-gray-500">Test acc</dt>
                <dd className="font-mono font-semibold text-gray-900">
                  0,8857 ± 0,0066
                </dd>
              </div>
            </dl>
            <p className="mt-3 text-xs text-gray-500">
              CV ile multi-seed farkı sadece 0,005 → veri varyansına karşı stabil.
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title={
              <span className="inline-flex items-center gap-2">
                <GitBranch className="h-4 w-4" /> Stack
              </span>
            }
          />
          <CardBody className="space-y-2 text-sm text-gray-700">
            <p>
              <b>Backend:</b> FastAPI + Uvicorn · transformers (PyTorch) · LIME · Pydantic
              v2
            </p>
            <p>
              <b>Frontend:</b> Vite · React 18 · TypeScript · Tailwind CSS · TanStack
              Query · React Router · Recharts
            </p>
            <div className="flex flex-wrap gap-1.5 pt-1">
              <Badge tone="primary">Python 3.10</Badge>
              <Badge tone="primary">PyTorch 2.5 + CUDA</Badge>
              <Badge tone="primary">React 18</Badge>
              <Badge tone="primary">TypeScript 5</Badge>
              <Badge tone="primary">Tailwind 3</Badge>
            </div>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader
          title={
            <span className="inline-flex items-center gap-2">
              <Users className="h-4 w-4" /> Ekip
            </span>
          }
        />
        <CardBody className="space-y-3 text-sm text-gray-700">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-gray-200 bg-gray-50 p-3">
              <p className="text-xs uppercase tracking-wide text-gray-500">Geliştirici</p>
              <p className="font-medium text-gray-900">Ahmet Dülger</p>
            </div>
            <div className="rounded-md border border-gray-200 bg-gray-50 p-3">
              <p className="text-xs uppercase tracking-wide text-gray-500">Geliştirici</p>
              <p className="font-medium text-gray-900">Muhammed Esat Demir</p>
            </div>
            <div className="rounded-md border border-gray-200 bg-gray-50 p-3 sm:col-span-2">
              <p className="inline-flex items-center gap-1 text-xs uppercase tracking-wide text-gray-500">
                <GraduationCap className="h-3.5 w-3.5" /> Danışman
              </p>
              <p className="font-medium text-gray-900">Dr. Öğr. Üyesi İlkay Yelmen</p>
            </div>
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={
            <span className="inline-flex items-center gap-2">
              <BookOpen className="h-4 w-4" /> Sürüm notları
            </span>
          }
        />
        <CardBody>
          <dl className="grid gap-3 text-sm sm:grid-cols-3">
            <div>
              <dt className="text-xs uppercase tracking-wide text-gray-500">Model</dt>
              <dd className="mt-0.5 font-mono text-gray-700">
                v3_genisletilmis · seed 43
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-gray-500">Pre-trained</dt>
              <dd className="mt-0.5 font-mono text-gray-700">
                dbmdz/bert-base-turkish-cased
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-gray-500">
                Demo sürümü
              </dt>
              <dd className="mt-0.5 font-mono text-gray-700">0.2.0</dd>
            </div>
          </dl>
        </CardBody>
      </Card>
    </div>
  );
}
