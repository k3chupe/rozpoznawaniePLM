# Etap 5 — Gesty dynamiczne z LSTM

**Cel:** Rozpoznawanie gestow wymagajacych ruchu (np. litery J, Z w PJM) przez analize sekwencji klatek za pomoca sieci LSTM.

**Czego sie nauczylem:**
- LSTM do klasyfikacji sekwencji czasowych
- Normalizacja czasowa wideo do stalej liczby klatek (30)
- Bufor deque do sledzenia "historii" klatek na zywo
- Logika czyszczenia buforu po wykryciu gestu

## Workflow

```powershell
.\venv\Scripts\Activate.ps1

# 1. Nagraj krotkie filmy gestow (trzymaj spacje podczas ruchu)
python nagrywanie_filmow_do_danych.py

# 2. Wytrenuj LSTM
python terning_na_filmie.py

# 3. Demo na zywo
python ruch_pokaz.py
```

## Pliki wejsciowe / wyjsciowe

| Plik | Opis |
|---|---|
| `plan_nagrania.txt` | Plan nagrywania: `LITERA ILOSC` per linia |
| `../nagrania_gestow/<LITERA>/*.mp4` | Nagrane filmy per klasa |
| `model_gesty_ruchome.keras` | Wytrenowany model LSTM |
| `etykiety_ruch.pkl` | Etykiety klas |

## Architektura modelu

```
Input(30, 64) -> LSTM(64, return_sequences=True) -> Dropout(0.2)
              -> LSTM(32) -> Dropout(0.2)
              -> Dense(32, relu)
              -> Dense(N_klas, softmax)
```

Kazda probka to sekwencja 30 klatek po 64 cechy (21 punktow 3D znormalizowanych + kat).

## Szczegoly implementacji

- `nagrywanie_filmow_do_danych.py` nagrywa czysty obraz (bez overlay'u MediaPipe), MediaPipe rysowany tylko na podgladu
- `terning_na_filmie.py` interpoluje sekwencje do 30 klatek niezaleznie od dlugosci nagrania
- `ruch_pokaz.py` czysci bufor po kazdym wykrytym gescie (pewnosc > 85%), co wymusza czekanie na kolejny pelny ruch

## Wniosek

LSTM skutecznie rozpoznaje gesty dynamiczne. Glowne wyzwanie to zebranie wystarczajacej ilosci zroznicowanych nagran per klasa.
