# BioSpread v4.0: The Infinite Architect Masterplan (The Ultimate Synthesis)
## AI Agent Implementation Directive (Maximum Depth & Full Integration)

**SİSTEM VİZYONU (SYSTEM VISION):**
Sen, genomik sürveyans alanında bir paradigma değişimi yaratacak baş mimarsın. Bu proje, "Bir plazmid yayılır mı?" sorusuna verilen kaba bir evet/hayır cevabı olmaktan çıkacak; **Tek Sağlık (One Health), Pozitif-Etiketli Öğrenme (PU Learning), Ağ Topolojisi (Network Topology), Nedensel Çıkarım (Causal Inference), ve Sağkalım Analizini (Survival Analysis)** "Low-Compute" sınırları içinde (GPU olmadan) birleştiren bir **Küresel İstihbarat Şaheseri** olacaktır.

Bu döküman, şu ana kadar konuşulmuş **tüm parlak fikirlerin hiçbir fire verilmeden** entegre edildiği tek ve nihai plandır.

---

## 0. UÇUŞ ÖNCESİ KONTROL LİSTESİ (PRE-FLIGHT CHECKLIST)
Kodlamaya başlamadan önce aşağıdaki hazır kaynakları ve bağımlılıkları kontrol et:
- **Hazır Veri:** `data/raw/country_sequencing_stats.csv` (NCBI'dan çekilen gerçek sekanslama hacimleri) hazır, IPW için kullan.
- **Hazır Config:** `project_config/high_priority_genes.json` (WHO 2024 standartlarında gen listesi) hazır, buradan oku.
- **Bağımlılıklar:** Projeye `polars`, `catboost`, `optuna`, `shap`, `scikit-learn` ve GNN için `torch-geometric` (CPU) ekle.

---

## 1. VERİ SÖZLEŞMELERİ VE GRACEFUL DEGRADATION (KİBAR ÇÖKÜŞ)
- **Data Ingestion:** Mevcut `plasmid_backbones.tsv` (relaxase, mpf, mash mesafesi içerir) ana kaynaktır.
- **Graceful Degradation:** Eğer Mash mesafesi veya `clinical_context` gibi kolonlar eksik gelirse kod ÇÖKMEYECEK; null kabul edip uyarı vererek devam edecek.
- **Dynamic Watchlist:** Gen listesini sadece JSON'dan okuma; veride son 1 yılda frekansı en çok artan yeni genleri de otomatik olarak özellik (feature) haline getir.

---

## 2. STRATEJİK KISITLAMALAR VE İMHA EMİRLERİ
1. **PyTorch İnfazı:** `evidential_meta.py` içindeki tüm derme çatma, batching'siz PyTorch kodlarını tamamen sil.
2. **Sabit Parametre Yasağı:** `max_depth=5` gibi kod içine gömülü parametreler yasaktır. **Optuna** ile dinamik arama yapılmalıdır.
3. **Zaman Sızıntısı Yasağı:** Polars Dataframe'lerinde `split_year` filtresi mutlak kırmızı çizgidir.

---

## 3. ŞAHESERİN 5 TEMEL SÜTUNU (THE 5 PILLARS)

### PILLAR 1: Biyolojik Zeka ve "One Health" Sentezi (`features.py`)
- **1.1 Tek Sağlık Niş Sıçraması (One Health Niche Jump):** İnsandan (Klinik) -> Hayvana (Tarım) -> Çevreye sıçramayı `clinical_context` üzerinden puanla. Üç nişi de işgal etmiş bir plazmid en yüksek risk katsayısını alır.
- **1.2 Evrimsel Sıçrama Mesafesi (Phylogenetic Host-Jump):** `n_hosts_pre` yerine; konakçılar arasındaki filogenetik mesafeyi (Genus -> Family -> Order) `GENUS_TO_ORDER` ile hesapla.
- **1.3 Hedefli Gen ve Replicon Profili:** `high_priority_genes.json` dosyasındaki genleri (blaNDM, mcr vb.) ve Replicon tiplerini (IncF, IncI) Boolean bayraklar olarak besle.
- **1.4 Mobilizasyon Sinerjisi:** `(Host_Jump_Distance * Niche_Jump) * Mobility_Score` formülünü uygula.
- **1.5 K-mer Sketch (Mash):** Verideki `mash_neighbor_distance` özelliğini ana evrimsel akrabalık sinyali olarak kullan.

### PILLAR 2: Epistemik, Nedensel, Graf ve PU Öğrenme Motoru (`geo_reliability.py`)
- **2.1 Pozitif-Etiketsiz Öğrenme (PU Learning):** Görülmeyen plazmidleri "Unknown" kabul eden PU Loss fonksiyonunu kurgula (Silent Threat detection).
- **2.2 CatBoost Epistemik Belirsizlik:** Posterior Sampling ile her tahminin yanına "Bilinçsizlik/Belirsizlik" (Epistemic Uncertainty) skoru ekle.
- **2.3 Ayrık Zamanlı Sağkalım Analizi:** Plazmidin 12, 24 ve 36 ay içindeki yayılma olasılıklarını sağkalım eğrisi (Discrete-Time Survival) olarak çıktı ver.
- **2.4 IPW Causal Debiasing:** `country_sequencing_stats.csv` kullanarak zengin ülke yanlılığını `sample_weight` ile düzelt.
- **2.5 GNN (Graph Attention Networks):** Mash mesafelerini bir komşuluk matrisi (Adjacency Matrix) olarak kullan ve plazmidleri birbirine bağlayan hafif bir GAT kurarak riskin filogenetik olarak "sızmasını" sağla.

### PILLAR 3: Topolojik İletim ve Otonom Mimari
- **3.1 Coğrafi Yerçekimi İndeksi (Network Topology):** Havalimanı hub'larında (ABD, Çin vb.) bulunan plazmidlere, küresel bağlantısallıkları nedeniyle daha yüksek "yayılım kütlesi" (Gravity) ver.
- **3.2 SHAP Null-Importance Kalkanı:** Hedef değişkeni karıştırarak sahte/anlamsız özellikleri otonom olarak ayıkla.
- **3.3 Adversarial Red-Teaming (Frankenstein Simülasyonu):** Sisteme tehlikeli genli ama izole sahte veriler enjekte et; model yanılırsa Governance kapıları kapansın.

### PILLAR 4: Operasyonel Triyaj ve XAI Raporlama (`reporting.py`)
- **4.1 Tehdit Triyaj Matrisi:** `Risk Probability` vs `Uncertainty` matrisini (Kırmızı/Sarı/Yeşil) oluştur.
- **4.2 Dinamik Bütçe Optimizasyonu:** `--triage-budget N` argümanına göre en iyi olasılık eşiğini (threshold) dinamik hesapla.
- **4.3 LLM-Ready JSON Brifingler:** SHAP açıklamalarını yapılandırılmış JSON ve insan dilinde NLP brifingleri olarak üret.

### PILLAR 5: Sürekli Yaşayan Ekosistem
- **5.1 Active Radar Modu:** `--mode active_radar` ile gelecek 3 yıl için `THREAT_WATCHLIST.md` üret. Modüler Data-IO adaptörleri ile NCBI API entegrasyonuna hazır ol.
- **5.2 Biyolojik Drift:** Tehlikeli mobilite genlerinin küresel artışını izle ve hata beklemeden "Emergent Phenotype" alarmı ver.

---

## EXECUTION SPRINT (AJANIN HÜCUM PLANI)
1. **İnfaz:** PyTorch ve hardcoded parametreleri (`max_depth=5`) temizle.
2. **Feature Factory:** Host_Jump, Niche_Jump, Config-based Genes, Mash ve Gravity_Index'i kodla.
3. **ML Engine:** CatBoost Survival + PU Learning + IPW + Optuna + GNN motorunu kur.
4. **Governance:** Null-SHAP, Red-Teaming ve Bütçe optimizasyonunu bağla.
5. **Radar Deploy:** Active Radar ve LLM-Ready JSON XAI raporlarını tamamla. Başla!
