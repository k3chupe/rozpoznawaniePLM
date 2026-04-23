# Etap 3 — XGBoost i porownanie modeli

**Cel:** Sprawdzenie czy XGBoost (drzewa gradientowe) bedzie lepszy od sieci MLP dla tego zadania. Wizualne porownanie obu modeli rownolegle.

**Czego sie nauczylem:**
- XGBoost dla klasyfikacji wieloklasowej (`multi:softprob`)
- `compute_sample_weight` dla balansowania klas w XGBoost
- Zapisywanie modelu XGBoost w formacie JSON

## Wymagania wstepne

Skopiuj do tego folderu modele z etapu 2:
- `model_gesty_punkty.keras`
- `etykiety_punkty.pkl`

## Workflow

```powershell
.\venv\Scripts\Activate.ps1

# 1. Wytrenuj XGBoost (dane z etapu 2 w ../lepsze_dane/)
python xgb_trenowanie.py

# 2. Porownanie obu modeli na zywo
python wizualizacja_porownanie.py

# 3. Porownanie na nagraniu (plik ../nagrania_testy/alfabet.mp4)
python nagranie_porownij.py
```

## Pliki wejsciowe / wyjsciowe

| Plik | Opis |
|---|---|
| `../lepsze_dane/*.jpg` | Dane treningowe (flat, nazwa: `LITERA_*.jpg`) |
| `model_gesty_xgboost.json` | Wytrenowany model XGBoost |
| `etykiety_xgboost.pkl` | Etykiety klas XGBoost |
| `../nagrania_testy/alfabet.mp4` | Nagranie do testu |
| `../nagrania_testy/alfabet_porownanie_nn_xgb.mp4` | Wynik porownania na nagraniu |

## Wniosek

Oba modele maja porownywalna dokladnosc na testowanych gestach. MLP nieco lepiej generalizuje dla trudniejszych przypadkow. Przejscie do etapu 4 — optymalizacja architektury i nowe API MediaPipe.
