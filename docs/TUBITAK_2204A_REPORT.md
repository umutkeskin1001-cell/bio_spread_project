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
7. Deployment aşamasında reverse-complement averaged inference kullanıldı.
8. Test, held-out test ve non-plasmid control setleriyle ölçüm yapıldı.

## Modelin Özgün Yönleri

- Eğitim en fazla 2,048 plasmid ile sınırlıdır.
- 501,526 parametreyle hızlı ve taşınabilir bir modeldir.
- Circular plasmid positional encoding, plasmidlerin dairesel biyolojisini modele yansıtır.
- Task-specific evidence windows, modelin hangi dizilim bölgelerine daha fazla önem verdiğini gösterir.
- Reverse-complement averaged inference, aynı plasmidin farklı yön gösterimlerinde daha kararlı sonuç üretir.

## Sonuçlar

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Ortalama Skor |
|---|---:|---:|---:|---:|
| Validation | 77.49% | 89.54% | 83.96% | 83.66% |
| Test | 72.78% | 87.70% | 72.86% | 77.78% |
| Held-out | 80.90% | 93.78% | 82.68% | 85.78% |

Non-plasmid stres özeti:

| Ölçüt | Değer |
|---|---:|
| False mobile rate | 4.11% |
| False AMR rate | 14.44% |
| False expansion rate | 11.11% |

## Yorum

AMR tahmini held-out sette güçlüdür. Expansion tahmini kullanılabilir seviyededir. Mobility sınıflandırması hâlâ en zor görevdir, ancak held-out sette artmıştır. Reverse-complement averaging, validation skorunu hafif düşürse de test ve held-out genellemesini yükselttiği için deployment açısından kabul edilmiştir. Proje sunumunda bu trade-off dürüstçe anlatılmalıdır.

## Etik ve Güvenlik

Model yalnızca araştırma ve eğitim amaçlı ön tarama aracıdır. Klinik, çevresel, regülasyon veya biyogüvenlik kararları için tek başına kullanılmamalıdır. Yüksek riskli sonuçlar deneysel veya anotasyon tabanlı yöntemlerle doğrulanmalıdır.

## Demo Akışı

1. Kullanıcı FASTA dizisini yükler.
2. Sistem DNA'yı normalize eder ve 56 pencere özelliği çıkarır.
3. Sistem orijinal ve reverse-complement gösterimleri birlikte değerlendirir.
4. Model mobility, AMR ve expansion risklerini üretir.
5. Web arayüzü risk skorlarını ve task-specific evidence window sıralamasını gösterir.
6. Sunumda modelin hızlı, kompakt, anotasyonsuz ve orientation-robust çalıştığı vurgulanır.
