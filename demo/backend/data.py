"""Statik metrikler ve örnekler.

Tüm sayılar `models/optimized_v3/*/summary_multiseed.csv`,
`models/optimized_v3/v3_genisletilmis/.../test_report.txt`,
`reports/xai_final/selected_examples.csv` dosyalarından elle alındı.

3-seed birleşik CM seed42+seed43+seed44 toplamıdır (n=4180); per-class P/R/F1
bu birleşik CM'den yeniden hesaplandı.
"""

OVERVIEW_METRICS = {
    "test_accuracy":  {"mean": 0.8857, "std": 0.0066},
    "macro_f1":       {"mean": 0.8797, "std": 0.0073},
    "weighted_f1":    {"mean": 0.8836, "std": 0.0073},
    "f1_negatif":     {"mean": 0.9047, "std": 0.0111},
    "f1_notr":        {"mean": 0.8056, "std": 0.0112},
    "f1_pozitif":     {"mean": 0.9286, "std": 0.0026},
    "test_size_per_seed": 1378,
    "test_size_combined_3seed": 4180,
    "dataset_size": 14337,
    "model": "dbmdz/bert-base-turkish-cased",
    "chosen_seed": 43,
    "labels": ["negatif", "nötr", "pozitif"],
}

CONFUSION_MATRIX_3SEED = {
    "labels": ["negatif", "nötr", "pozitif"],
    "matrix": [
        [1228,   81,    2],
        [ 165,  964,  147],
        [  11,   72, 1510],
    ],
    "support": {"negatif": 1311, "nötr": 1276, "pozitif": 1593},
    "total": 4180,
    "note": "Sum of seed 42, 43, 44 confusion matrices on their respective test sets.",
}

PER_CLASS_METRICS_3SEED = {
    "labels": ["negatif", "nötr", "pozitif"],
    "precision": {"negatif": 0.8746, "nötr": 0.8630, "pozitif": 0.9102},
    "recall":    {"negatif": 0.9367, "nötr": 0.7555, "pozitif": 0.9479},
    "f1":        {"negatif": 0.9046, "nötr": 0.8057, "pozitif": 0.9287},
    "support":   {"negatif": 1311,   "nötr": 1276,   "pozitif": 1593},
}

MULTI_MODEL_RESULTS = {
    "metric_keys": ["accuracy", "macro_f1", "weighted_f1", "f1_negatif", "f1_notr", "f1_pozitif"],
    "models": [
        {
            "key": "berturk",
            "name": "BERTurk (dbmdz/bert-base-turkish-cased)",
            "highlighted": True,
            "metrics": {
                "accuracy":    {"mean": 0.8857, "std": 0.0066},
                "macro_f1":    {"mean": 0.8797, "std": 0.0073},
                "weighted_f1": {"mean": 0.8836, "std": 0.0073},
                "f1_negatif":  {"mean": 0.9047, "std": 0.0111},
                "f1_notr":     {"mean": 0.8056, "std": 0.0112},
                "f1_pozitif":  {"mean": 0.9286, "std": 0.0026},
            },
        },
        {
            "key": "electra_tr",
            "name": "ELECTRA-tr (dbmdz/electra-base-turkish-cased-discriminator)",
            "highlighted": False,
            "metrics": {
                "accuracy":    {"mean": 0.8906, "std": 0.0130},
                "macro_f1":    {"mean": 0.8853, "std": 0.0143},
                "weighted_f1": {"mean": 0.8891, "std": 0.0131},
                "f1_negatif":  {"mean": 0.9094, "std": 0.0186},
                "f1_notr":     {"mean": 0.8143, "std": 0.0243},
                "f1_pozitif":  {"mean": 0.9321, "std": 0.0016},
            },
        },
        {
            "key": "xlmr",
            "name": "XLM-RoBERTa (xlm-roberta-base)",
            "highlighted": False,
            "metrics": {
                "accuracy":    {"mean": 0.8709, "std": 0.0101},
                "macro_f1":    {"mean": 0.8658, "std": 0.0099},
                "weighted_f1": {"mean": 0.8696, "std": 0.0109},
                "f1_negatif":  {"mean": 0.8973, "std": 0.0056},
                "f1_notr":     {"mean": 0.7882, "std": 0.0152},
                "f1_pozitif":  {"mean": 0.9119, "std": 0.0134},
            },
        },
        {
            "key": "mbert",
            "name": "mBERT (bert-base-multilingual-cased)",
            "highlighted": False,
            "metrics": {
                "accuracy":    {"mean": 0.8538, "std": 0.0036},
                "macro_f1":    {"mean": 0.8462, "std": 0.0051},
                "weighted_f1": {"mean": 0.8514, "std": 0.0046},
                "f1_negatif":  {"mean": 0.8657, "std": 0.0072},
                "f1_notr":     {"mean": 0.7609, "std": 0.0169},
                "f1_pozitif":  {"mean": 0.9120, "std": 0.0036},
            },
        },
    ],
}

ABLATION_RESULTS = {
    "metric_keys": ["accuracy", "macro_f1", "f1_notr"],
    "stages": [
        {
            "key": "baseline",
            "label": "v3_baseline",
            "description": "Kaggle dahil, ham 13.586 satır (tek seed)",
            "metrics": {
                "accuracy":   {"mean": 0.9161, "std": None},
                "macro_f1":   {"mean": 0.9122, "std": None},
                "f1_notr":    {"mean": 0.8579, "std": None},
            },
            "n_seeds": 1,
            "note": "Tek seed, eski test seti. Karşılaştırılabilir değil ama referans olarak verildi.",
        },
        {
            "key": "real_test",
            "label": "v3_real_test",
            "description": "Sadece gerçek yorumlar, ilk filtre",
            "metrics": {
                "accuracy":   {"mean": 0.8966, "std": 0.0057},
                "macro_f1":   {"mean": 0.8329, "std": 0.0134},
                "f1_notr":    {"mean": 0.6245, "std": 0.0347},
            },
            "n_seeds": 3,
        },
        {
            "key": "real_test_fixed",
            "label": "v3_real_test_fixed",
            "description": "Etiket düzeltmeleri sonrası",
            "metrics": {
                "accuracy":   {"mean": 0.8963, "std": 0.0089},
                "macro_f1":   {"mean": 0.8303, "std": 0.0163},
                "f1_notr":    {"mean": 0.6156, "std": 0.0382},
            },
            "n_seeds": 3,
        },
        {
            "key": "bigtest",
            "label": "v3_bigtest",
            "description": "Genişletilmiş test seti, daha sıkı değerlendirme",
            "metrics": {
                "accuracy":   {"mean": 0.8679, "std": 0.0066},
                "macro_f1":   {"mean": 0.8377, "std": 0.0069},
                "f1_notr":    {"mean": 0.6852, "std": 0.0107},
            },
            "n_seeds": 3,
        },
        {
            "key": "genisletilmis",
            "label": "v3_genisletilmis",
            "description": "AI ile dengeli veri seti, nihai model",
            "metrics": {
                "accuracy":   {"mean": 0.8857, "std": 0.0066},
                "macro_f1":   {"mean": 0.8797, "std": 0.0073},
                "f1_notr":    {"mean": 0.8056, "std": 0.0112},
            },
            "n_seeds": 3,
            "highlighted": True,
        },
    ],
}

DATASET_DISTRIBUTION = {
    "total": 14337,
    "by_class": {
        "pozitif": 5307,
        "nötr":    4660,
        "negatif": 4370,
    },
    # 2026-05-09 sonrası: neredekal (40) → otelpuan_otelyorum altında birleştirildi
    # (Hocanın geri bildirimi A.1). Final veri seti `_v3_v2.csv`'de 3 kaynak.
    "by_source": {
        "otelpuan_otelyorum": 11086,  # doğal kullanıcı yorumu (otelpuan 11.046 + neredekal 40)
        "ai_generated":        2500,  # yapay zekâ üretimli — yalnız train, test'e sızdırılmadı
        "manuel_toplama":       751,  # elle-yazımlı (yapı-temelli) — AI üretimi DEĞİL
    },
    "source_descriptions": {
        "otelpuan_otelyorum": "Doğal kullanıcı yorumu (otelpuan.com kazımalarından)",
        "ai_generated":       "Yapay zekâ üretimli (yalnız eğitim setinde)",
        "manuel_toplama":     "Elle-yazımlı yapı-temelli koleksiyon (AI üretimi DEĞİL)",
    },
    "splits": {"train": 11781, "val": 1178, "test": 1378},
}

EXAMPLES = [
    {
        "id": 1,
        "kind": "dogru_negatif",
        "label": "Doğru negatif",
        "text": "Sadece casino ya önem verilmiş herşey dahil gidilmesine rağmen birçok şeyin ücreti var çok saçma bir düzen.",
        "true": "negatif",
        "pred": "negatif",
        "correct": True,
        "wc": 17,
    },
    {
        "id": 2,
        "kind": "dogru_notr",
        "label": "Doğru nötr",
        "text": "Etkinlik olmaması iyi değildi. Çok üzün süre kalınmaz. Her şey dahil konseptine alıştık sanırım o yüzden. Olumlu yönü olarak temiz bir tesisti. Personel yaklaşımı iyi.",
        "true": "nötr",
        "pred": "nötr",
        "correct": True,
        "wc": 25,
    },
    {
        "id": 3,
        "kind": "dogru_pozitif",
        "label": "Doğru pozitif",
        "text": "genel olarak otelden çok memnun kaldık. personel güler yüzlü ve yardımsever. yemekler lezzetli. içecekler ana binada kaliteli.",
        "true": "pozitif",
        "pred": "pozitif",
        "correct": True,
        "wc": 17,
    },
    {
        "id": 4,
        "kind": "hata_notr_negatif",
        "label": "Hata: nötr → negatif",
        "text": "Otelde havuz ve gösteri imkanları gayet güzeldi lakin yemek lezzetleri çok kötüydü kahvaltı çeşitleri, pizza buzdolabı kokuyordu. Oda temizliği her gün yapıldı memnunduk. Otel kesinlikle aile çocuk oteli herkesin çocuğu var sevgililerin baş başa gitmesini önermem kalabalık bir otel.",
        "true": "nötr",
        "pred": "negatif",
        "correct": False,
        "wc": 39,
    },
    {
        "id": 5,
        "kind": "hata_notr_pozitif",
        "label": "Hata: nötr → pozitif",
        "text": "Otel her şeyiyle güzel. Temizdi. Bakımlıydı. Bir tek casionlar sıkıntı sizin için önemli değilse casinoya çok takılmayın ama benim fikrimi çok etkiledi.",
        "true": "nötr",
        "pred": "pozitif",
        "correct": False,
        "wc": 22,
    },
    {
        "id": 6,
        "kind": "hata_pozitif_notr",
        "label": "Hata: pozitif → nötr",
        "text": "Otelin olanakları güzeldi ve çok temizdi. Hava kötüydü, bu yüzden olanaklardan yararlanamadım. Ancak düzenli ve yemeklerle ilgili herhangi bir sıkıntı yoktu.",
        "true": "pozitif",
        "pred": "nötr",
        "correct": False,
        "wc": 21,
    },
    {
        "id": 7,
        "kind": "hata_negatif_notr",
        "label": "Hata: negatif → nötr",
        "text": "Güzel bir otel. Her şey standarttı. Havuzu kirliydi, yaprak dökülüyor rüzgar olduğu için o da normaldi. Kahvaltıları kötüydü, ürünleri bayattı, akşam yemeği de fena değildi. Odaları da temizdi. Fiyat ve performans açısından orta derece ne pahalı ne de ucuz.",
        "true": "negatif",
        "pred": "nötr",
        "correct": False,
        "wc": 39,
    },
    {
        "id": 8,
        "kind": "dogru_notr_uzun",
        "label": "Doğru nötr (uzun)",
        "text": "Yol üstü bir duraktı, beklentim düşüktü. Oda temiz, yatak rahat. İnternet sadece lobide çekiyor. Personel az ama çalışkan. Kahvaltı tabağı doyurucuydu. Girişteki resepsiyon yoğunluğu nedeniyle biraz bekledim. Klima kumandası bozuktu, manuel açtım. Genel anlamda fena değildi.",
        "true": "nötr",
        "pred": "nötr",
        "correct": True,
        "wc": 36,
    },
]

ERROR_SUMMARY = {
    "total_test": 1378,
    "total_errors": 148,
    "error_rate": 0.1074,
    "errors_by_direction": {
        "nötr_to_pozitif":   47,
        "nötr_to_negatif":   41,
        "negatif_to_nötr":   31,
        "pozitif_to_nötr":   28,
        "pozitif_to_negatif": 1,
    },
    "errors_by_category": {
        "guvenli_yanlis":             95,
        "orta_seviye":                26,
        "belirsiz_dengesiz":          24,
        "etiket_gurultusu_supheli":    3,
    },
    "confidence_stats": {
        "mean_max_prob":          0.832,
        "median_max_prob":        0.8479,
        "high_conf_errors_count": 98,
        "low_conf_errors_count":  3,
    },
    "wc_stats": {"mean": 50.5, "median": 39, "min": 3, "max": 263},
}


# =============================================================================
# 5-FOLD CROSS-VALIDATION (BERTurk + ELECTRA-tr, 15+15 = 30 koşu)
# Kaynak: models/cv/v3_genisletilmis_5fold_{berturk,electra}/summary_cv.json
# =============================================================================
CV_RESULTS = {
    "protocol": "AI-aware StratifiedKFold, n_splits=5, 3 seed (42/43/44)",
    "n_runs_per_model": 15,
    "n_folds": 5,
    "n_seeds": 3,
    "test_size_per_fold": 2367,
    "models": [
        {
            "key": "berturk",
            "name": "BERTurk",
            "elapsed_hours": 4.42,
            "metrics": {
                "accuracy":   {"mean": 0.9035, "std": 0.0055},
                "macro_f1":   {"mean": 0.8750, "std": 0.0066},
                "weighted_f1":{"mean": 0.9042, "std": 0.0052},
                "f1_negatif": {"mean": 0.9284, "std": 0.0064},
                "f1_notr":    {"mean": 0.7496, "std": 0.0128},
                "f1_pozitif": {"mean": 0.9472, "std": 0.0043},
            },
            "highlighted": True,
        },
        {
            "key": "electra_tr",
            "name": "ELECTRA-tr",
            "elapsed_hours": 17.19,
            "metrics": {
                "accuracy":   {"mean": 0.9062, "std": 0.0056},
                "macro_f1":   {"mean": 0.8782, "std": 0.0076},
                "weighted_f1":{"mean": 0.9068, "std": 0.0056},
                "f1_negatif": {"mean": 0.9295, "std": 0.0076},
                "f1_notr":    {"mean": 0.7553, "std": 0.0156},
                "f1_pozitif": {"mean": 0.9498, "std": 0.0041},
            },
            "highlighted": False,
        },
    ],
    "comparison": {
        "delta_macro_f1": 0.0032,
        "delta_f1_notr":  0.0057,
        "speed_ratio":    3.9,
        "interpretation": (
            "ELECTRA tüm metriklerde nominal olarak az miktarda üstün, "
            "ancak σ ile aynı düzeyde fark → istatistiksel olarak ayırt edilemez. "
            "BERTurk 3,9× daha hızlı eğitiliyor."
        ),
    },
    "vs_multi_seed": {
        "note": (
            "CV ile multi-seed (BERTurk Macro-F1 0,8797±0,0073) farkı sadece 0,005 "
            "→ model hem init varyansı hem veri varyansı altında kararlı."
        ),
    },
}


# =============================================================================
# McNEMAR İSTATİSTİKSEL ANLAMLILIK TESTLERİ
# Multi-model: reports/mcnemar/multi_model/mcnemar_summary.json
# CV-fold:     reports/mcnemar/cv_fold/mcnemar_cv_summary.json
# =============================================================================
MCNEMAR_SUMMARY = {
    "total_tests": 33,
    "bert_electra_tests": 18,
    "multi_model": {
        "n_tests": 18,
        "n_pairs": 6,
        "n_seeds_per_pair": 3,
        "bert_vs_electra": {
            "min_p_value": 0.0298,
            "median_p_value": 0.7404,
            "max_p_value": 1.0000,
            "bonferroni_m3_threshold": 0.0167,
            "bonferroni_m18_threshold": 0.00278,
            "significant_m3": False,
            "significant_m18": False,
            "interpretation": "Eşdeğer (her iki Bonferroni şemasında da)",
        },
        "bert_vs_mbert": {
            "min_p_value": 0.00008,
            "median_p_value": 0.0005,
            "significant_m3": True,
            "significant_m18": True,
            "interpretation": "Türkçe-spesifik mBERT'e karşı üstün (sağlam)",
        },
        "bert_vs_xlmr": {
            "min_p_value": 0.00453,
            "median_p_value": 0.0761,
            "significant_m3": True,
            "significant_m18": False,
            "interpretation": "Şemaya bağlı kararsız (sadece seed 44 anlamlı)",
        },
    },
    "cv_fold": {
        "n_tests": 15,
        "comparison": "BERTurk-CV vs ELECTRA-CV (fold-paired)",
        "min_p_value": 0.0759,
        "median_p_value": 0.3626,
        "max_p_value": 0.9354,
        "n_significant_uncorrected": 0,
        "n_significant_bonferroni": 0,
        "fishers_combined_p": 0.3683,
        "interpretation": (
            "0/15 fold anlamlı, Fisher's combined p=0,368 → "
            "BERT-ELECTRA CV ortamında da eşdeğer (bağımsız doğrulama)."
        ),
    },
    "akademik_sonuc": (
        "BERT-ELECTRA çiftine yönelik 18 McNemar testi (3 multi-model + 15 CV-fold) "
        "hiçbiri Bonferroni anlamlı değil → eşdeğerlik bulgusu çok sağlam."
    ),
}


# =============================================================================
# KLASİK ML BASELINE (TF-IDF + LogReg/LinearSVC)
# Kaynak: reports/classical_baseline/summary.json
# =============================================================================
CLASSICAL_BASELINE = {
    "n_seeds": 3,
    "n_models": 3,
    "n_total_runs": 9,
    "models": [
        {
            "key": "tfidf_char_wb_logreg",
            "name": "TF-IDF char_wb (3-5) + LogReg",
            "metrics": {
                "macro_f1":   {"mean": 0.7850, "std": 0.0028},
                "f1_notr":    {"mean": 0.6471, "std": 0.0080},
                "accuracy":   {"mean": 0.8043, "std": 0.0018},
            },
            "best_classical": True,
        },
        {
            "key": "tfidf_word_linearsvc",
            "name": "TF-IDF word (1-2) + LinearSVC",
            "metrics": {
                "macro_f1":   {"mean": 0.7823, "std": 0.0031},
                "f1_notr":    {"mean": 0.6322, "std": 0.0133},
                "accuracy":   {"mean": 0.8043, "std": 0.0004},
            },
        },
        {
            "key": "tfidf_word_logreg",
            "name": "TF-IDF word (1-2) + LogReg",
            "metrics": {
                "macro_f1":   {"mean": 0.7787, "std": 0.0095},
                "f1_notr":    {"mean": 0.6277, "std": 0.0245},
                "accuracy":   {"mean": 0.8009, "std": 0.0054},
            },
        },
    ],
    "vs_berturk": {
        "berturk_macro_f1": 0.8797,
        "best_classical_macro_f1": 0.7850,
        "delta_macro_f1": 0.0947,
        "delta_f1_notr": 0.1585,
        "interpretation": (
            "BERTurk +9,5 puan Macro-F1, +15,8 puan F1-nötr üstün. "
            "Literatür beklentisiyle (8-12 puan) birebir uyumlu."
        ),
    },
    "top_features_per_class": {
        "negatif": [
            ("kötü", 4.55), ("kötüydü", 3.75), ("değildi", 3.36),
            ("yetersiz", 3.11), ("hiç", 2.90),
        ],
        "nötr": [
            ("ama", 6.00), ("ancak", 5.91), ("fakat", 5.61),
            ("fena", 2.26), ("ortalama", 2.13),
        ],
        "pozitif": [
            ("güzeldi", 4.98), ("her", 4.13), ("iyiydi", 3.89),
            ("güzel", 3.45), ("voyage", 3.30),
        ],
    },
    "xai_cross_validation": (
        "Nötr top-3 katsayı (ama 6,00 / ancak 5,91 / fakat 5,61) BERTurk LIME "
        "bulgusunu (§4.10) bağımsız olarak doğruladı: discourse marker = nötrlük "
        "ilişkisi BERTurk-spesifik artefakt değil, veri setinin yapısal özelliği."
    ),
}


# =============================================================================
# SARKAZM/İRONİ XAI (8 sentetik sarkastik örnek)
# Kaynak: reports/irony_xai/summary.json
# =============================================================================
SARCASM_ANALYSIS = {
    "n_total": 8,
    "n_sarcastic": 7,
    "n_control_direct_negative": 1,
    "n_correct": 2,
    "n_sarcastic_correct": 1,
    "n_sarcastic_wrong": 6,
    "control_correct": True,
    "avg_confidence_when_wrong": 0.916,
    "interpretation": (
        "7 sarkastik örnekten 6'sı yanlış (yüksek güvenle, p>0,7); "
        "kontrol grubu doğru → model temel görevi yapabiliyor, sınırlama "
        "spesifik olarak sarkazma yöneliktir. LIME ile yüzeysel pozitif kelime "
        "bağımlılığı kanıtlandı (muhteşemdi:+0,57, güleryüzlüydü:+0,34 vb.)."
    ),
    "examples": [
        {
            "id": 1,
            "text_short": "Hayatımda gördüğüm en harika otel, banyoda sıcak su yokmuş...",
            "true": "negatif",
            "pred": "nötr",
            "confidence": 0.714,
            "ironic_signal": "harika, mükemmel — pozitif kelimeler",
        },
        {
            "id": 4,
            "text_short": "Otel kahvaltısı muhteşemdi, üç çeşit reçel ve tost makinesi vardı...",
            "true": "negatif",
            "pred": "pozitif",
            "confidence": 0.994,
            "ironic_signal": "muhteşem, lüks → +0,570 LIME katkısı",
        },
        {
            "id": 5,
            "text_short": "Odaya girer girmez küf kokusu beni karşıladı...",
            "true": "negatif",
            "pred": "negatif",
            "confidence": 0.991,
            "ironic_signal": "Tek doğru: somut negatif kelimeler baskın",
        },
        {
            "id": 6,
            "text_short": "Eh işte, beklediğim gibi vasat bir oteldi...",
            "true": "negatif",
            "pred": "negatif",
            "confidence": 0.977,
            "ironic_signal": "Kontrol grubu (sarkazm DEĞİL) — model doğru",
        },
    ],
}


# =============================================================================
# SIZINTI AUDIT (3 multi-seed + 15 CV-fold = 18 bağımsız split)
# Kaynak: reports/leakage_audit/v3_genisletilmis_seed{42,43,44}/leakage_summary.json
#         reports/leakage_audit/v3_genisletilmis_5fold_berturk/leakage_summary.json
# =============================================================================
LEAKAGE_AUDIT = {
    "method": "3-yöntemli bağımsız audit (Exact + Normalized + Jaccard)",
    "multi_seed": [
        {"seed": 42, "exact_tt": 0, "normalized_tt": 8, "jaccard_080_tt": 23, "jaccard_090_tt": 11},
        {"seed": 43, "exact_tt": 0, "normalized_tt": 8, "jaccard_080_tt": 16, "jaccard_090_tt": 6},
        {"seed": 44, "exact_tt": 0, "normalized_tt": 6, "jaccard_080_tt": 26, "jaccard_090_tt": 14},
    ],
    "multi_seed_aggregate": {
        "exact_invariant_zero": True,
        "j090_mean": 10.3,
        "j090_std": 4.2,
    },
    "cv_15_fold": {
        "n_folds": 15,
        "exact_invariant_zero": True,
        "normalized_mean": 7.1,
        "normalized_std": 2.3,
        "j090_mean": 10.7,
        "j090_std": 2.1,
        "j090_min": 8,
        "j090_max": 14,
    },
    "impact_bound": {
        "multi_seed_test_size": 1393,
        "cv_test_size": 2367,
        "max_impact_macro_f1_multi_seed": 0.008,
        "max_impact_macro_f1_cv": 0.0059,
        "interpretation": (
            "Etki bound'u σ ile aynı düzeyde (≤ 0,008 puan); "
            "manuel_toplama yapı-temelli koleksiyonun şablon-varyantları kaynaklı."
        ),
    },
}


# =============================================================================
# EDA + PEARSON KORELASYON
# Kaynak: reports/eda/eda_summary.json + reports/eda/corr_matrix.csv
# =============================================================================
EDA_SUMMARY = {
    "n_meta_features": 10,
    "key_correlations": {
        "rating_auto_x_sentiment_id": {
            "pearson_r": 0.8127,
            "interpretation": "Etiketleme tutarlılığının deterministik kanıtı",
        },
        "is_ai_x_is_notr": {
            "point_biserial_r": 0.662,
            "interpretation": "AI'ların %100'ü nötr (kategorik determinism)",
        },
        "is_ai_x_wc": {
            "pearson_r": -0.345,
            "interpretation": "AI yorumları kısa (medyan 6 vs gerçek 32 kelime)",
        },
        "pos_kw_x_neg_kw_overall": {
            "pearson_r": 0.021,
            "interpretation": "Simpson Paradoksu artefakt'ı (sınıf-içi r=+0,1-0,2)",
        },
    },
    "class_distribution": {
        "negatif": {"count": 4370, "ratio": 0.305, "wc_median": 39},
        "nötr":    {"count": 4660, "ratio": 0.325, "wc_median": 8},
        "pozitif": {"count": 5307, "ratio": 0.370, "wc_median": 32},
        "total":   14337,
    },
    "source_distribution": {
        "otelpuan_otelyorum": {"count": 11086, "ratio": 0.773, "type": "doğal kullanıcı yorumu"},
        "ai_generated":       {"count": 2500,  "ratio": 0.174, "type": "AI üretimli (sadece train)"},
        "manuel_toplama":     {"count": 751,   "ratio": 0.052, "type": "elle-yazımlı yapı-temelli"},
    },
}
