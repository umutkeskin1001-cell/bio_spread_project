# TÜBİTAK 2204-A Proje Raporu Taslağı

## Proje Adı

Cassiopeia Prime: DNA Dizisinden Plasmid Hareketlilik, Antimikrobiyal Direnç ve Yayılım Riski Tahmini

## Amaç

Bu projenin amacı, yalnızca plasmid DNA dizisini kullanarak üç biyolojik riski hızlı ve kompakt bir modelle tahmin etmektir:

- plasmidin mobilizasyon sınıfı,
- antimikrobiyal direnç genleri taşıma olasılığı,
- geniş coğrafi yayılım riski.

Model BLAST, gen anotasyonu veya metadata gerektirmez. Böylece düşük kaynaklı ortamlarda hızlı ön tarama yapılabilir.

## Yöntem

1. PLSDB kaynaklı plasmid dizileri filtrelendi.
2. Benzer plasmidlerin farklı splitlere sızmasını azaltmak için grup-aware split uygulandı.
3. Her dizi 56 çok ölçekli pencereye ayrıldı: 32 kısa, 16 orta, 8 uzun pencere.
4. Her pencereden ters-komplement tutarlı k-mer özellikleri ve yapısal DNA özellikleri çıkarıldı.
5. Cassiopeia Prime modeli bu özelliklerden üç görevi birlikte öğrendi.
6. Validation set üzerinde kalibrasyon uygulandı.
7. İki model versiyonu (v14 ve v15) ağırlıklı ortalama ile ensemble edildi.
8. Deployment aşamasında reverse-complement averaged inference kullanıldı.
9. Test, held-out test ve non-plasmid control setleriyle ölçüm yapıldı.

## Modelin Özgün Yönleri

- Eğitim en fazla 2,048 plasmid ile sınırlıdır.
- 568,437 parametreyle hızlı ve taşınabilir bir modeldir.
- Circular plasmid positional encoding, plasmidlerin dairesel biyolojisini modele yansıtır.
- Task-specific evidence windows, modelin hangi dizilim bölgelerine daha fazla önem verdiğini gösterir.
- Reverse-complement averaged inference, aynı plasmidin farklı yön gösterimlerinde daha kararlı sonuç üretir.
- Model ensemble (v14+v15) ile tek modele göre +1.14 puan task score iyileştirmesi sağlanmıştır.

## Sonuçlar

Cassiopeia Prime v0.3.0 (şampiyon: v14+v15 ensemble) — ölçülen denetimli sonuçlar:

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Ortalama Skor |
|---:|---:|---:|---:|---:|
| Validation | 77.63% | 91.99% | 88.54% | 86.05% |
| Test | 74.18% | 92.16% | 87.43% | 84.59% |
| Held-out | **78.33%** | **93.89%** | **88.78%** | **87.00%** |

### Held-out Per-class Mobility

| Sınıf | F1 | Recall |
|---:|---:|---:|
| non-mobilizable (0) | 0.741 | 71.5% |
| **mobilizable (1)** | **0.725** | **75.6%** |
| conjugative (2) | 0.878 | 87.8% |

## Yorum

Nihai model (v14+v15 ensemble) held-out test setinde **%87.00 task score** elde etmiştir. Mobilizable sınıfı F1 değeri başlangıçtaki 0.678'den **0.725'e** yükseltilmiş, recall %64.7'den %75.6'ya çıkarılmıştır. Bu iyileştirme per-class weight, mobility-specific focal loss (γ=0.5), balanced sampling'de mobility ağırlığının arttırılması ve v14+v15 ensemble kombinasyonuyla sağlanmıştır.

AMR AUROC %93.89 ile çok güçlü seviyededir. Expansion AUROC da %88.78'e yükselmiştir. Mobility sınıflandırmasında conjugative (F1=0.878) ve non-mobilizable (F1=0.741) sınıfları güçlü biçimde ayrılmaktadır. L-BFGS sıcaklık + bias kalibrasyonu ile tüm görevlerde ECE düşük seviyededir.

Proje sunumunda bu başarılar ve 2,048-plasmid veri sınırı dürüstçe anlatılmalıdır.

## Etik ve Güvenlik

Model yalnızca araştırma ve eğitim amaçlı ön tarama aracıdır. Klinik, çevresel, regülasyon veya biyogüvenlik kararları için tek başına kullanılmamalıdır. Yüksek riskli sonuçlar deneysel veya anotasyon tabanlı yöntemlerle doğrulanmalıdır.

## Demo Akışı

1. Kullanıcı FASTA dizisini yükler.
2. Sistem DNA'yı normalize eder ve 56 pencere özelliği çıkarır.
3. Sistem orijinal ve reverse-complement gösterimleri birlikte değerlendirir.
4. İki model (v14+v15) ağırlıklı ortalama ile ensemble edilir.
5. Model mobility, AMR ve expansion risklerini üretir.
6. Web arayüzü risk skorlarını ve task-specific evidence window sıralamasını gösterir.
7. Sunumda modelin hızlı, kompakt, anotasyonsuz ve orientation-robust çalıştığı vurgulanır.
