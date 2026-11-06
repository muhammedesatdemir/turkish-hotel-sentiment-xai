# -*- coding: utf-8 -*-
"""
IEEE Template_fixed.docx'i temizleyip bildiri icerigini template'in KENDI stilleriyle
doldurur. Boylece 2-sutun IEEE formati ve stiller korunur.

NOT: Icerik su an TURKCE (kullanici Ingilizceye cevirecek). Sekil/tablo yer tutuculari
metin olarak isaretlenir; gorseller manuel veya basit add_picture ile eklenir.
"""
from pathlib import Path
import copy
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
TMPL = ROOT / "dosyalar" / "Template_fixed.docx"
OUT = ROOT / "dosyalar" / "IEEE_Bildiri_AhmetDulger_MuhammedEsatDemir.docx"
FIG1 = ROOT / "reports" / "ieee_figurleri" / "fig1_pipeline.png"
FIG2 = ROOT / "reports" / "ieee_figurleri" / "fig2_lime.png"

doc = Document(str(TMPL))

# --- Mevcut gövdedeki paragraf/tabloları sil; section yapısını yeniden kuracağız ---
body = doc.element.body
final_sectPr = None  # belgenin en sonundaki sectPr (sayfa boyutu/marj burada)
for child in list(body):
    if child.tag.endswith("}sectPr"):
        final_sectPr = copy.deepcopy(child)
    body.remove(child)


def make_cols(sectPr_el, num, space):
    """sectPr içindeki cols ayarını set et."""
    cols = sectPr_el.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr_el.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), space)


# Final sectPr'i 2 SÜTUN yap (gövdenin ana düzeni)
if final_sectPr is None:
    final_sectPr = OxmlElement("w:sectPr")
make_cols(final_sectPr, 2, "18pt")
# Final sectPr'i şimdilik body'ye EKLEME — içerik eklendikten sonra eklenecek.

# Başlık/yazar bölümünü TEK SÜTUN tutmak için, yazar bloğundan sonra
# araya bir "continuous section break" koyacağız. Bunun için bir yardımcı:
def insert_section_break_after_last_paragraph(num_cols, space):
    """Son paragrafın pPr'ine sectPr ekleyerek o noktaya kadar olan bölümü ayırır."""
    last_p = doc.paragraphs[-1]._p
    pPr = last_p.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        last_p.insert(0, pPr)
    sect = OxmlElement("w:sectPr")
    typ = OxmlElement("w:type"); typ.set(qn("w:val"), "continuous"); sect.append(typ)
    c = OxmlElement("w:cols"); c.set(qn("w:num"), str(num_cols)); c.set(qn("w:space"), space)
    sect.append(c)
    pPr.append(sect)


def style_ok(name):
    try:
        _ = doc.styles[name]
        return True
    except KeyError:
        return False


def add(text, style=None):
    """sectPr'den ÖNCE paragraf ekle ki sütun ayarı en sonda kalsın."""
    p = doc.add_paragraph(text)
    if style and style_ok(style):
        p.style = doc.styles[style]
    return p


def add_pic(path, width_in=3.2):
    p = doc.add_paragraph()
    try:
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width_in))
    except Exception as e:
        p.add_run(f"[Görsel: {Path(path).name}]")
    return p


# ============ İÇERİK ============
# Başlık
add("A Leakage-Free and Explainable Sentiment Analysis Approach for Turkish Hotel Reviews", "papertitle")

# Yazar bloğu (placeholder)
add("1st Given Name Surname — dept., Istinye University, Istanbul, Turkey — email\n"
    "2nd Given Name Surname — dept., Istinye University, Istanbul, Turkey — email\n"
    "3rd Given Name Surname (Advisor) — dept., Istinye University, Istanbul, Turkey — email",
    "Author")

# Başlık+yazar bloğu TEK SÜTUN (full-width) kalsın → buraya section break koy.
# Bu break'ten önceki içerik 1 sütun, sonrası 2 sütun olur.
insert_section_break_after_last_paragraph(1, "36pt")

# Abstract
add("Abstract—Online hotel reviews are a strategic data source for measuring customer "
    "satisfaction in tourism; however, the rich morphology of Turkish and reviews with mixed "
    "sentiment make automatic sentiment analysis challenging. This paper presents a three-class "
    "(negative/neutral/positive) sentiment analysis system for Turkish hotel reviews. An original "
    "dataset of 14,337 reviews compiled from multiple sources is built, and a BERTurk model is "
    "fine-tuned with a Weighted Cross Entropy loss. The main contribution is the experimental "
    "demonstration that the leakage of synthetic (AI-generated) samples into the test set "
    "artificially inflates performance metrics, and the proposal of an evaluation protocol "
    "(ai_only_in_train) that keeps synthetic samples only in the training set so that the test set "
    "consists of 100% real user reviews. On a three-seed average, the model reaches 88.57% accuracy "
    "and 0.8797 Macro-F1, a +9.5-point improvement over a classical baseline. Decisions are "
    "explained with LIME and SHAP; the Turkish adversative conjunction 'ama' (but) is shown to "
    "positively contribute to the neutral class.   [NOT: Bu Abstract Ingilizce yazildi; istersen "
    "Turkce tasvirden kendi cevirini koy.]", "Abstract")

add("Keywords—sentiment analysis, Turkish NLP, BERTurk, explainable AI, LIME, SHAP, data leakage, transformer", "Keywords")

# I. INTRODUCTION
add("Introduction", "Balk1")
add("[TR] Çevrimiçi platformlarda üretilen kullanıcı yorumlarının hızla artması, metinsel verileri "
    "turizm sektöründe stratejik bir kaynak hâline getirmiştir [1]. Otel yorumları hizmet kalitesinin "
    "değerlendirilmesi açısından kritik olsa da elle incelenmeleri ölçeklenebilir değildir; bu nedenle "
    "otomatik duygu analizi karar destek sistemlerinin vazgeçilmez bir bileşenidir.", "GvdeMetni")
add("[TR] Türkçe için duygu analizi iki temel zorluk içerir. Birincisi, Türkçenin eklemeli yapısı ve "
    "zengin morfolojisidir [2]. İkincisi, üç sınıflı yapılarda nötr sınıfın tanımsal belirsizliği nedeniyle "
    "en düşük performansı göstermesidir. Buna ek olarak, sentetik veya makine çevirisi içeren verilerin "
    "test setine sızması (test-set contamination) metrikleri yapay olarak yükseltir.", "GvdeMetni")
add("[TR] Bu makalenin başlıca katkıları: (i) üç bağımsız yöntemle sızıntı denetiminden geçirilmiş "
    "14.337 yorumluk özgün bir Türkçe veri seti; (ii) sentetik sızıntının metrikleri şişirdiğini gösteren "
    "ve bunu engelleyen bir değerlendirme protokolü (ai_only_in_train); (iii) LIME ve SHAP ile model "
    "kararlarının açıklanması ve Türkçe karşıtlık bağlaçlarının nötr sınıfa katkısının gösterilmesi.", "GvdeMetni")

# II. RELATED WORK
add("Related Work", "Balk1")
add("[TR] Türkçe Duygu Analizi ve Transformer Modeller. Derin öğrenmenin yükselişiyle Transformer "
    "tabanlı mimariler baskın hâle gelmiştir [3]. Türkçe korpuslar üzerinde ön-eğitilen BERTurk [4] "
    "dil-özel görevlerde üstün sonuçlar verir; XLM-RoBERTa [5] ve ELECTRA [6] de yaygındır. Bu modellerin "
    "Türkçe duygu analizinde karşılaştırmalı değerlendirilmesi sınırlıdır.", "GvdeMetni")
add("[TR] Açıklanabilir Yapay Zekâ. LIME [7] yerel doğrusal yaklaşımlarla etkili kelimeleri belirler; "
    "SHAP [8] Shapley değerlerine dayanır. Bu yöntemlerin Türkçe sentiment modellerine uygulanması, "
    "dilbilimsel kavrayışı doğrulamak ve güveni artırmak açısından önemlidir.", "GvdeMetni")
add("[TR] Veri Sızıntısı ve Sentetik Veri. Sentetik veri artırımı sınıf dengesizliğini azaltır ancak "
    "test setine karışması metrik şişmesine yol açabilir. Bu sorun literatürde yeterince ele alınmamıştır; "
    "bu çalışma deneysel bir kanıt ve pratik bir çözüm sunar.", "GvdeMetni")

# III. METHODOLOGY
add("Methodology", "Balk1")
add("[TR] Önerilen sistemin uçtan uca akışı Şekil 1'de gösterilmektedir.", "GvdeMetni")
add_pic(FIG1, 3.2)
add("Fig. 1. End-to-end research pipeline.", "figurecaption")

add("A. Data Collection and Consolidation", "Balk2")
add("[TR] Nihai veri seti 14.337 özgün Türkçe otel yorumundan oluşur (Tablo I): 11.086 gerçek kullanıcı "
    "yorumu (%77,3), 2.500 kalite-filtrelenmiş sentetik nötr örnek (%17,4) ve karşıtlık bağlaçları "
    "hedeflenerek elle yazılmış 751 nötr yorum (%5,2). Makine çevirisi içeren bir kaynak, düşük kalite ve "
    "sızıntı riski nedeniyle çıkarılmıştır.", "GvdeMetni")

add("B. Preprocessing and Labeling", "Balk2")
add("[TR] Metinler UTF-8'e normalize edilmiş, NFKC uygulanmış ve birebir tekrarlar kaldırılmıştır. "
    "Üç sınıflı etiketleme kullanıcı puanlarına dayalı eşik kuralıyla yapılmış; etiket güvenilirliği "
    "Pearson korelasyonuyla doğrulanmıştır (r = +0,81).", "GvdeMetni")

add("C. Leakage-Free Data Splitting", "Balk2")
add("[TR] Temel metodolojik katkı budur. Azınlık nötr sınıfının test temsilini güçlendirmek için "
    "sınıfa özgü test oranları uygulanmıştır (nötr %20, diğerleri %10). ai_only_in_train protokolü ile "
    "sentetik örnekler yalnızca eğitime konmuş, test seti %100 gerçek yorumdan oluşturulmuştur. Test "
    "güvenilirliği exact-match (0 sızıntı), normalize eşleşme ve Jaccard yakın-duplicate ile denetlenmiştir; "
    "etki ≤ 0,008 puandır.", "GvdeMetni")

add("D. Model and Loss Function", "Balk2")
add("[TR] Nihai model dbmdz/bert-base-turkish-cased üzerine kurulmuştur. Oversampling ezberlemeyi "
    "artırdığından Weighted Cross Entropy (WCE) tercih edilmiş; ağırlıklar sınıf frekanslarıyla ters orantılı "
    "hesaplanmıştır. Model üç tohumla (42, 43, 44) eğitilmiş; max_len 256, lr 2e-5, 5 epoch.", "GvdeMetni")

add("E. Explainability", "Balk2")
add("[TR] Kararlar LIME [7] ve SHAP [8] ile açıklanmış; kutupsal kelimelerde tutarlı sonuçlar elde "
    "edilmiştir.", "GvdeMetni")

# IV. EXPERIMENTS AND RESULTS
add("Experiments and Results", "Balk1")
add("A. Experimental Setup", "Balk2")
add("[TR] Eğitimler NVIDIA RTX 3060 GPU, FP16 ve deterministik tohumlama ile yapılmıştır. Performans "
    "üç-tohum ortalaması üzerinden raporlanmıştır. Klasik bir taban model (TF-IDF + Lojistik Regresyon) "
    "referanstır.", "GvdeMetni")

add("B. Main Results", "Balk2")
add("[TR] Nihai BERTurk modeli %88,57 doğruluk ve 0,8797 Macro-F1'e ulaşmıştır (Tablo I). Per-sınıf F1: "
    "negatif 0,9047, nötr 0,8056, pozitif 0,9286. Klasik tabana (0,785) kıyasla +9,5 puan üstünlük "
    "sağlanmıştır.", "GvdeMetni")
add("[TABLO I BURAYA — TABLE I. FINAL MODEL PERFORMANCE AND BASELINE COMPARISON]", "GvdeMetni")

add("C. Effect of Synthetic Data Leakage", "Balk2")
add("[TR] Sentetik örnekler tüm bölmelere karıştırıldığında Macro-F1 yapay olarak 0,9122'ye yükselir. "
    "Bu, sentetik örneklerin ezberlenmesinden kaynaklanan bir metrik sızıntısıdır. Sentetik örnekler "
    "yalnızca eğitime hapsedildiğinde skor gerçekçi düzeye (0,8797) iner. Bu, yüksek bir metriğin her zaman "
    "güvenilir olmadığını gösterir.", "GvdeMetni")

add("D. Multi-Model Comparison", "Balk2")
add("[TR] Dört Transformer mimarisi aynı protokolde karşılaştırılmıştır (Tablo II). ELECTRA-tr nominal "
    "olarak en yüksek Macro-F1'i (0,8853) verse de varyansı iki kat yüksektir (σ = 0,0143); fark McNemar "
    "testleriyle anlamsız bulunmuş (Bonferroni sonrası p > 0,0167) ve 5-katlı CV ile teyit edilmiştir. Bu "
    "nedenle nihai model BERTurk seçilmiştir. Çok dilli mimariler geride kalmıştır.", "GvdeMetni")
add("[TABLO II BURAYA — TABLE II. MULTI-MODEL COMPARISON]", "GvdeMetni")

add("E. Explainability Findings", "Balk2")
add("[TR] Şekil 2 bir örnek için LIME katkılarını gösterir. Model 'çok saçma bir düzen' yorumunu negatif "
    "sınıflandırırken en yüksek katkıyı 'saçma' kelimesi (+0,50) sağlamıştır. Daha dikkat çekici olan, "
    "karşıtlık bağlacı 'ama'nın nötr sınıfa +0,10 katkı sağlamasıdır. SHAP, LIME ile tutarlıdır. Hataların "
    "%99'u nötr ekseninde toplanmış; pozitif-negatif karışması tek örnekte (%0,7) görülmüştür.", "GvdeMetni")
add_pic(FIG2, 3.2)
add("Fig. 2. LIME word contributions for a sample review.", "figurecaption")

# V. CONCLUSION
add("Conclusion", "Balk1")
add("[TR] Bu çalışmada Türkçe otel yorumları için sızıntıdan arınmış ve açıklanabilir bir duygu analizi "
    "sistemi sunulmuştur. ai_only_in_train protokolü metrik şişmesini engelleyerek güvenilir ölçüm "
    "sağlamıştır. Model klasik tabana kıyasla +9,5 puan üstünlük göstermiş; LIME/SHAP modelin Türkçe "
    "karşıtlık bağlaçlarını anlamlı kullandığını ortaya koymuştur. Sınırlamalar: domain bağımlılığı ve "
    "sarkazm zafiyeti. Gelecek çalışmalar: etiketleyiciler-arası uyum, GroupKFold ve domain adaptasyonu.", "GvdeMetni")

# ACKNOWLEDGMENT
add("Acknowledgment", "Balk1")
add("[TR/EN] The authors thank their advisor for guidance throughout this study. [Placeholder]", "GvdeMetni")

# REFERENCES
add("References", "Balk1")
refs = [
 'B. A. Sparks and V. Browning, "Complaining in cyberspace: The motives and forms of hotel guests\' complaints online," J. Hosp. Mark. Manag., vol. 19, no. 7, pp. 797–818, 2010, doi: 10.1080/19368623.2010.508010.',
 'E. Demirtas and M. Pechenizkiy, "Cross-lingual polarity detection with machine translation," in Proc. 2nd Int. Workshop Issues Sentiment Discovery Opinion Mining (WISDOM), 2013, art. no. 9, pp. 1–8, doi: 10.1145/2502069.2502078.',
 'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. NAACL-HLT, 2019, pp. 4171–4186, doi: 10.18653/v1/N19-1423.',
 'S. Schweter, "BERTurk — BERT models for Turkish," Zenodo, 2020, doi: 10.5281/zenodo.3770924.',
 'A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," in Proc. ACL, 2020, pp. 8440–8451, doi: 10.18653/v1/2020.acl-main.747.',
 'K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning, "ELECTRA: Pre-training text encoders as discriminators rather than generators," in Proc. ICLR, 2020, arXiv:2003.10555.',
 'M. T. Ribeiro, S. Singh, and C. Guestrin, "\'Why should I trust you?\': Explaining the predictions of any classifier," in Proc. 22nd ACM SIGKDD, 2016, pp. 1135–1144, doi: 10.1145/2939672.2939778.',
 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. NeurIPS, vol. 30, 2017, pp. 4765–4774.',
]
for i, r in enumerate(refs, 1):
    add(f"[{i}] {r}", "references")

# Gövdenin (Abstract → References) 2 sütun olması için final sectPr'i en sona ekle.
body.append(final_sectPr)

doc.save(str(OUT))
print("IEEE docx yazildi:", OUT)
print("Paragraf:", len(doc.paragraphs), "| Gorsel:", len(doc.inline_shapes))
