# Tespit Edilen Sorunlar ve Düzeltme Planı

## Analiz Bulguları

### 1. Web Arayüzü Sorunları (web/app.js)
- **Kritik**: Evidence ağırlıkları `Math.random()` ile üretiliyor. "Deterministik PRNG" yorumu var ama kullanılmıyor. Yarışma sunumunda sorunlu.
- **Kritik**: `snapshot_date: "2026-06-01"` yanlış tarih (bugün 2026-06-04)
- `BENCHMARK_DATA` içinde yorumlar yanıltıcı

### 2. utils.py Mantık Hataları
- `predict_batch`: Evidence scores'lar sadece ilk pass'ten (orijinal yön) alınıyor. RC averaging'de mobilite olasılıkları ortalansa bile evidence ortalama alınmıyor.
- `expected_calibration_error`: Son bin'de `(hi < 1)` koşulu, ama `p == 1.0` (lo, hi=0.99, 1.0) edge case'inde doğru ama kafa karıştırıcı
- `DNASequenceAugmentation`: Her zaman yeni liste döndürüyor, test `result == records` yanlış

### 3. api.py Güvenlik/UX
- **CORS middleware yok**: Web arayüzü ayrı host'taysa çalışmaz
- Input validation iyi ama "sequence_id" injection korunmasız (ama pydantic field_validator var)

### 4. model.py Konfig
- `CassiopeiaConfig`'e `aux_loss_weight` eklenmemiş ama yaml'da kullanılıyor → sessizce yok sayılıyor

### 5. cli.py Sağlamlık
- `predict_cmd` try/except yok, bir hata tüm batch'i durdurur

### 6. docs/TUBITAK_2204A_REPORT.md Tutarsızlık
- "Reverse-complement averaging, validation skorunu hafif düşürse de" → çelişkili

### 7. pyproject.toml
- Üst sınırlar (>=8,<10) → minor

## Düzeltme Planı
- [x] web/app.js: Math.random() → deterministik LCG PRNG
- [x] web/app.js: snapshot_date güncelle
- [x] utils.py: predict_batch evidence averaging fix
- [x] api.py: CORS middleware ekle
- [x] model.py: aux_loss_weight dataclass'a ekle
- [x] cli.py: predict try/except ekle
- [x] utils.py: augmentation identity check fix
- [x] TÜBİTAK raporu: validation düşüşü ifadesi düzelt
- [x] utils.py: window_dropout mask == False noqa kaldır
- [x] utils.py: expected_calibration_error son bin edge case
- [x] Son test çalıştırması
