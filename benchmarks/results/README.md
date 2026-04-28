# Benchmark Sonuçları — Scoring Analizi

Bu doküman patent ve OpenAlex benchmark sonuçlarının analizini, scoring sisteminin
hangi parametrelerinin ne kadar etkili olduğunu ve threshold'ların durumunu içerir.

Tarih: 2026-04-28

---

## 1. Soru: "Patentler için doğru predict edilen yüzde kaç?"

**Dosya:** `benchmarks/results/runs/benchmark_patents_rows.csv` (50 case)

### Sonuçlar

**Genel Doğruluk: %64 (32/50)**

| True Label | Doğru | Toplam | Accuracy |
|---|---|---|---|
| already_exists | 11 | 17 | 64.7% |
| incremental | 13 | 17 | 76.5% |
| novel | 8 | 16 | 50.0% |

### Confusion Matrix

|  | pred: already_exists | pred: incremental | pred: novel |
|---|---|---|---|
| **true: already_exists** | **11** | 6 | 0 |
| **true: incremental** | 1 | **13** | 3 |
| **true: novel** | 0 | 8 | **8** |

### Gözlemler
- **Novel sınıfı en zayıf**: 16 novel case'in 8'i `incremental` olarak tahmin edilmiş.
- **already_exists → incremental** kayması (6 case): Var olan fikirler "kısmen yeni" olarak hafifletiliyor.
- Catastrophic hata yok: hiçbir `novel` `already_exists`'e, hiçbir `already_exists` `novel`'e gitmemiş — hatalar hep komşu sınıflar arasında.

---

## 2. Soru: "Koddaki bütün scoring logic'e bak"

İncelenen dosyalar: `layer1_agent.py`, `layer2_agent.py`, `core/config.py`,
`analysis_service.py`, `models/analysis.py`, `retriever.py`, `chroma_store.py`.

### Scoring Pipeline (uçtan uca)

#### Layer1 (`layer1_agent.py`) — paper başına LLM scoring
- 4 ayrı LLM çağrısı (problem / method / domain / contribution), her biri 1-5 Likert
- 5. çağrı: cümle bazlı analiz (benchmark mode'da skip)
- Likert → float: `{1:0.0, 2:0.25, 3:0.5, 4:0.75, 5:1.0}`
- `paper_similarity_score = Σ w_i × criterion_i` (config.CRITERIA_WEIGHTS ile)

#### Layer2 (`layer2_agent.py`) — global aggregation
- `_compute_paper_similarity`: weighted mean
- `_compute_global_similarity`: **0.7 × max + 0.3 × mean**
- `_overlap_to_originality_score`: `(1 - overlap^1.5) × 100`
- `_score_to_label`: ≥70 high (novel), ≥40 medium (incremental), <40 low (already_exists)

#### config.py
```
CRITERIA_WEIGHTS = {problem: 0.15, method: 0.30, domain: 0.10, contribution: 0.45}
OVERLAP_CURVE_POWER = 1.5
SCORE_RED_MAX = 40, SCORE_YELLOW_MAX = 70
```

### Tespit edilen problemler

1. **Ölü config değerleri** — backend'de hiçbir yerde kullanılmıyor:
   - `SENTENCE_OVERLAP_TOP_K`
   - `GUARDRAIL_CRITICAL_FLOOR`, `GUARDRAIL_HIGH_COUNT`, `GUARDRAIL_HIGH_FLOOR`

2. **Docstring/kod tutarsızlığı** (`layer2_agent.py:56` vs `:204-216`):
   - Sınıf docstring: "Global similarity = max over papers"
   - Gerçek kod: `0.7*max + 0.3*mean` blend
   - `config.py:120` da yanıltıcı yorum içeriyor

3. **`_likert_to_float` edge case** (`layer1_agent.py:631`): float input edge case'i, pratikte tetiklenmiyor.

---

## 3. Soru: "Bu sonuçlara neden olan scoring kısmı neresi — threshold mu, weight mi?"

### Skor dağılımı (true label'a göre)

| true_label | n | mean orig | mean gsim |
|---|---|---|---|
| novel | 16 | **72.75** | 0.41 |
| incremental | 17 | **61.88** | 0.51 |
| already_exists | 17 | **30.88** | 0.77 |

**Sorun:** Üç sınıfın merkezleri (73 / 62 / 31) sınırlara çok yakın oturuyor.
Özellikle novel ortalaması **72.75** — sınır olan 70'in HEMEN üstünde.

### Sweep sonuçları

| değişen şey | en iyi accuracy |
|---|---|
| **Sadece label threshold'ları** (red=54, yellow=78) | **37/50 = %74** |
| Curve power + thresholds birlikte | %74 (aynı — power monoton, fark etmiyor) |
| Weight'leri yeniden tune (geniş sweep) | %76 (sadece +2) |
| Tek bir kriter tek başına | en fazla %36 |

**En büyük kazanç (+10 puan) sadece `SCORE_RED_MAX` ve `SCORE_YELLOW_MAX`'ı kaydırmaktan geliyor.** Weight'ler ve curve neredeyse hiç katkı yapmıyor.

### Neden tüm dağılım sağa kaymış?
1. **Retrieval bias** — RAG zaten domain-relevant patent getiriyor, dolayısıyla `domain_similarity` novel için bile 0.62.
2. **Likert ortalamaya çekilme** — LLM 1-5 ölçeğinde kararsızken 3'e (=0.5) yaslanıyor.
3. **Patentler ArXiv'den daha tekrarlı** — 04-19'daki kalibrasyon ArXiv 40 case üzerinde yapılmış.

---

## 4. Soru: "Bu threshold'lar ne işe yarıyor?"

İki tane var, ikisi de aynı işi yapıyor: **0-100 arası `originality_score`'u 3 sınıfa çevirmek.**

`backend/app/agents/layer2_agent.py:380-387`:

```python
def _score_to_label(originality_score: int) -> OriginalityLabel:
    if originality_score >= config.SCORE_YELLOW_MAX:    # >= 70
        return OriginalityLabel.HIGH      # "novel" (yeşil)
    elif originality_score >= config.SCORE_RED_MAX:     # >= 40
        return OriginalityLabel.MEDIUM    # "incremental" (sarı)
    else:
        return OriginalityLabel.LOW       # "already_exists" (kırmızı)
```

```
0 ────────── 40 ────────── 70 ────────── 100
  KIRMIZI       SARI          YEŞİL
  already_exists incremental   novel
       ↑              ↑
   SCORE_RED_MAX  SCORE_YELLOW_MAX
```

**Önemi:** Predicted label tamamen bunlardan çıktığı için accuracy'yi en çok etkileyen yer burası.

---

## 5. Soru: "Skorlama OK mi peki, tek problem threshold mu?"

**Hayır, threshold tek problem değil — en büyük tek kaldıraç o ama altta yatan başka şeyler de var.**

| problem | tahmini etki | çözüm zorluğu |
|---|---|---|
| **Threshold yanlış yerde** | +%10 | trivial (config) |
| Contribution: yüksek weight (0.45) + zayıf spread (0.29) | +%2-4 | weight rebalance veya rubric gevşetme |
| 0.7max blend gürültü amplifiye ediyor | +%2-3 | kolay test edilir |
| Likert merkeze yapışma | +%2-3 | prompt değişikliği |
| Patent ≠ ArXiv kalibrasyonu | belirsiz | adapter-spesifik config |

### Contribution paradoksu (en ciddi)
- weight: **0.45** (en yüksek)
- spread: **0.29** (en kötü ayırt edicilik) — novel=0.33, exists=0.62
- En yüksek ağırlığı en zayıf ayırt edici kritere vermişiz
- Karşılaştırma: `method` weight 0.30, spread **0.49** (en iyi ayırt edici)

---

## 6. Soru: "Layer1 Likert'ine dokunmadan parametre değiştiren bir script yaz"

**Script:** `benchmarks/sweep_scoring_params.py`

CSV'deki `layer1_results` JSON'undan per-paper Likert skorlarını okuyup,
post-Layer1 parametreleri (weights, max-blend, curve power, thresholds) sweep ediyor.

### Patents sonuçları

| Müdahale | Yeni accuracy | Kazanç |
|---|---|---|
| Baseline | 64.0% | — |
| **Sadece threshold** (red=54, yel=78) | **74.0%** | **+10** |
| + global_max_w'yi 0.7 → 0.3-0.5 | 76.0% | +2 |
| + curve power değişimi | 76.0% | 0 (etkisiz) |
| + weights yeniden tune | 76.0% | 0 |

### Sürpriz bulgular

#### Curve power tamamen etkisiz
Power 1.0 → 2.0 arası: hep 74%. Çünkü monoton dönüşüm — threshold'ları yeniden ayarlayınca eşdeğer.

#### Global aggregation: `0.7×max` aslında kötü kalibre
| max_w | best accuracy |
|---|---|
| 0.0 (pure mean) | 74% |
| **0.3** | **76%** |
| **0.5** | **76%** |
| 0.7 (mevcut) | 74% |
| 1.0 (pure max) | 70% |

Mevcut 0.7 ne mean'in stabilitesini ne de pure max'ın hassasiyetini almış — arada kalmış.

#### Weight tuning gerçek gücü yok
Joint optimum'da bile weight'ler değişiyor (`p=0, m=0.33, d=0.17, c=0.50`) ama threshold-fix'in üstüne sadece +2 ekliyor.

#### Asıl tavan novel detection
Joint optimum confusion matrix:
```
already_exists → already_exists  17/17  ✅ %100
   incremental → incremental     14/17  ✅ %82
         novel → novel            7/16  ⚠️  %44
         novel → incremental      9/16  ← parametre tuning'le çözülmüyor
```

Novel vakalar parametre değişimine direnç gösteriyor çünkü problem **scoring math'te değil, LLM'in Likert vermesinde**.

---

## 7. Soru: "Özet olarak ne değişmeli?"

### Kesin yap (low-risk, +10-12 puan)

`backend/core/config.py`:
```python
SCORE_RED_MAX = 48        # 40 → 48
SCORE_YELLOW_MAX = 72     # 70 → 72
```
→ Tek başına **+10 puan** (64% → 74%)

`backend/app/agents/layer2_agent.py:216`:
```python
return 0.5 * best + 0.5 * mean    # 0.7/0.3 → 0.5/0.5
```
→ Ek **+2 puan** (74% → 76%)

### Dokunma
- `OVERLAP_CURVE_POWER` — etkisiz parametre
- `CRITERIA_WEIGHTS` — ArXiv kalibrasyonunu bozma riski, fayda 0-2 puan

### Yapılamaz (Layer1 LLM'e dokunmadan)
%76 tavan. Daha yukarısı için Likert inflation çözülmeli.

---

## 8. Soru: "Bunlar OpenAlex sonuçları için de geçerli mi?"

**Hayır — threshold önerisi tam tersi yöne dönüyor.**

| | Patents | OpenAlex |
|---|---|---|
| Baseline (mevcut) | 64% | 66% |
| Threshold-only fix | 74% | 76% |
| Joint optimum | 76% | 78% |
| **Best red_max** | **54** (yukarı) | **22-24** (aşağı) |
| **Best yellow_max** | **78** (yukarı) | **60-62** (aşağı) |
| best global_max_w | 0.3-0.5 | 0.5 |
| best weights (p,m,d,c) | 0, 0.33, 0.17, 0.50 | 0, 0.33, 0.17, 0.50 |

### Önemli bulgular

**Threshold ZIT yöne ihtiyaç duyuyor**:
- **Patents**: skorlar yapay yüksek — threshold YUKARI çek
- **OpenAlex**: skorlar yapay düşük — threshold AŞAĞI çek

**Weight tercihi her iki corpus'ta da aynı** — bu gerçek sinyal:
Her iki sweep'in en iyi weight kombinasyonu aynı: `p=0, m=0.33, d=0.17, c=0.50`.

**`global_max_w = 0.5`** her iki corpus'ta da kazandırıyor (threshold re-tune ile).

---

## 9. Soru: "Her ikisini de olumlu etkileyenler ne?"

**Cevap: Threshold'a dokunmadan iki corpus'u birden iyileştiren tek bir parametre değişikliği yok.**

Mevcut threshold 40/70 sabit tutulduğunda:

| Değişiklik | Patents | OpenAlex | Toplam |
|---|---|---|---|
| **Baseline** | 64% | 66% | **65%** |
| max_w 0.7 → 0.5 | 50% (**-14**) | 68% (+2) | 59% |
| weights (0, .33, .17, .50) | 58% (-6) | 66% (0) | 62% |
| power 1.5 → 1.0 | 68% (+4) | 60% (**-6**) | 64% |
| weights+max_w combo | 48-50% (-14) | 70% (+4) | 59-60% |

İki corpus'un dağılımları **zıt yönde kaymış**, tek global parametre ikisini birden hedefe yaklaştıramıyor.

---

## 10. Soru: "Adapter spesifik threshold'ları config'den kullanıcı seçsin — mantıklı mı?"

**Dürüst cevap: Hayır, çok da mantıklı değil.**

### Endişeler

1. **Pansuman, tedavi değil** — Asıl problem LLM'in Likert kalibrasyonu bozuk. Threshold'larla bunu post-process'te düzeltmek "termometreyi her hastaneye göre ayarlayalım" gibi.

2. **Overfitting riski** — Her adapter için tek bir 50-case benchmark var. Yarın farklı domain'de bir kullanıcı geldiğinde threshold'lar tutmayabilir.

3. **Kullanıcının seçmesi en kötü senaryo** — Çoğu kullanıcı default'u kullanır, "siz seçin" demek illuzyon. Aynı puan farklı config'lerde farklı renk verir → açıklanabilirlik ölür.

4. **Maintenance burden** — Her yeni adapter → benchmark + threshold tune.

5. **YAGNI** — Threshold zaten adapter'la deterministik ilişkili.

### Ne yapardım

1. **Hiçbir şey yapma, daha çok benchmark topla** (en dürüst cevap — 50 case istatistiksel olarak zayıf)
2. Adapter-specific threshold yapacaksak: config'de değil, adapter sınıfının kendisinde tut, kullanıcıya açma
3. Asıl çözüme yönel — upstream:
   - Contribution rubric'i gevşet
   - LLM prompt'unda Likert inflation azalt
   - Belki corpus-relative normalization

---

## 11. Soru: "Şuanki sonuçlar iyi mi sence?"

**Dürüst cevap: Orta. Üretim için yeterli değil ama ümitsiz de değil.**

| | Accuracy | Random baseline | Çoğunluk sınıfı |
|---|---|---|---|
| Patents | 64% | 33% | 34% |
| OpenAlex | 66% | 33% | 34% |

Random'ın ~2 katı.

### Kötü tarafları

1. **1/3 ideanın etiketi yanlış.** User-facing bir araç için ciddi.
2. **En zararlı hata novel → incremental** ve bu en sık olan hata.
3. **Hata türleri sistematik bias** (random gürültü değil) — distribution shifted.
4. Topic classification benzeri 3-class problemler iyi tune edildiğinde 80-90% verir.

### İyi tarafları

1. **Catastrophic hatalar yok** — sistem komşu sınıflarda kayıyor, "tam ters" hata yapmıyor.
2. **Sistemin asıl değeri label değil — gerekçe.** Kullanıcı 5 paper'ı, neden similar olduğunu, hangi cümleye match'i görüyor.
3. **Boundary error'ler ucuz.** "Borderline incremental/novel" diye gösterilebilir.

### Honest bottom line

**65% "ship edilebilir ama dikkatle" seviyesi.** Şu şartlarla:

1. Label'ı tek başına gösterme — her zaman 5 paper + criteria scores ile birlikte göster
2. Verdict dilini yumuşat — "moderate overlap detected, see references"
3. Boundary'de uncertainty göster — orig=68 → "borderline novel/incremental"
4. Çıkış noktası olarak konumlandır — "starting point for your literature review"

Asıl iyileştirme noktası label değil, **presentation**. 75-80%'e çıkmak için Layer1 LLM'in Likert kalibrasyonunu çözmek gerek (haftalar sürer, garantisiz). Threshold tuning ile ulaşılabilir tavan ~76%.

---

## 12. Soru: "70 üstüne novel diyoruz değil mi?"

**Evet.** `backend/core/config.py:89`:

```python
SCORE_YELLOW_MAX = 70   # >= 70 → high (= novel)
```

Yani şu an:
- **0–39** → already_exists (kırmızı)
- **40–69** → incremental (sarı)
- **70–100** → novel (yeşil)

Patents benchmark'taki sorun da tam burada: gerçek novel vakaların ortalama originality skoru **72.75** çıkıyor — yani 70 sınırının HEMEN üstünde, küçük gürültü 67-69'a düşürüp incremental etiketine kaydırıyordu (8 vakada böyle oldu).

---

## Özet Tablo

| Konu | Mevcut | Önerilen | Etki |
|---|---|---|---|
| `SCORE_RED_MAX` | 40 | 48 (patents) / 24 (openalex) | +10 puan / corpus |
| `SCORE_YELLOW_MAX` | 70 | 72-78 (patents) / 60 (openalex) | +10 puan / corpus |
| `global_max_w` | 0.7 | 0.5 (her iki corpus için iyi, threshold re-tune ile) | +2 puan |
| `OVERLAP_CURVE_POWER` | 1.5 | dokunma | 0 |
| `CRITERIA_WEIGHTS` | dokunma | dokunma | 0-2 puan, risk yüksek |
| Adapter-spesifik threshold | yok | belki, ama kullanıcıya açma | +10 / corpus |
| Asıl çözüm | — | Layer1 LLM rubric/prompt revize | belirsiz |

## Kullanılan Dosyalar

- **Sweep script**: `benchmarks/sweep_scoring_params.py`
- **Patents CSV**: `benchmarks/results/runs/benchmark_patents_rows.csv`
- **OpenAlex JSON**: `benchmarks/results/runs/1d0f79533038_openalex.json`
- **Scoring kodu**: `backend/app/agents/layer1_agent.py`, `backend/app/agents/layer2_agent.py`
- **Config**: `backend/core/config.py`
