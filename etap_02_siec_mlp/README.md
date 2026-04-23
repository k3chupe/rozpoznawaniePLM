# Etap 2 — Siec neuronowa MLP

**Cel:** Zastapienie dopasowania szablonow uczona siecia MLP (Multi-Layer Perceptron).

**Czego sie nauczylem:**
- Ekstrakcja 43 cech (21 punktow 2D znormalizowanych + kat orientacji)
- Augmentacja przez odbicia lustrzane
- Wagi klas (class weights) dla niezbalansowanych danych
- Callbacki Kerasa: EarlyStopping, ModelCheckpoint

## Workflow

```powershell
.\venv\Scripts\Activate.ps1

# 1. Zbierz zdjecia
python robienie_zdjec.py

# 2. Wytrenuj model
python trenowanie.py

# 3. Przetestuj na zywo
python pokazanie.py
```

## Pliki wejsciowe / wyjsciowe

| Plik | Opis |
|---|---|
| `../lepsze_dane/*.jpg` | Zdjecia treningowe (nazwa: `LITERA_timestamp.jpg`) |
| `plan_zbierania.txt` | Plan zbierania: `LITERA ILOSC` per linia |
| `model_gesty_punkty.keras` | Wytrenowany model (tworzony przez trenowanie.py) |
| `etykiety_punkty.pkl` | Etykiety klas (tworzony przez trenowanie.py) |

## Architektura modelu

```
Input(43) -> Dense(256, relu) -> Dropout(0.3)
          -> Dense(128, relu) -> Dropout(0.2)
          -> Dense(64, relu)
          -> Dense(N_klas, softmax)
```

## Wniosek

Siec MLP radzi sobie znacznie lepiej niz dopasowanie szablonow. Przetestowano rowniez XGBoost (etap 3). Przejscie do etapu 3 aby porownac oba podejscia.
