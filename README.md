# Rozpoznawanie Gestow Jezyka Migowego (PLM)

Projekt budowany iteracyjnie — kazdy folder `etap_*` to osobny eksperyment z wlasnym srodowiskiem wirtualnym i zaleznosciami.

## Struktura projektu

```
rozpoznawaniePLM/
├── setup_venv.bat                        # Skrypt tworzacy venv dla kazdego etapu
│
├── etap_01_dopasowanie_szablonow/        # Etap 1: dopasowanie geometryczne do wzorcow
├── etap_02_siec_mlp/                     # Etap 2: siec neuronowa MLP na zdjeciach
├── etap_03_xgboost_porownanie/           # Etap 3: XGBoost + porownanie z Keras
├── etap_04_keras_tuner/                  # Etap 4: optymalizacja hiperparametrow (Keras Tuner)
└── etap_05_lstm_ruch/                    # Etap 5: rozpoznawanie gestow dynamicznych (LSTM)
```

## Dane (wspoldzielone miedzy etapami)

Foldery z danymi sa wspoldzielone i leza w korzeniu repozytorium (ignorowane przez git):

| Folder | Opis | Uzywany w etapach |
|---|---|---|
| `alfabet/` | Zdjecia wzorcowe `wzor_*.jpg` | 01 |
| `do_nauki/` | Automatycznie zbierane klatki | 01 |
| `lepsze_dane/` | Recznie zebrane zdjecia (flat lub podfoldery) | 02, 03, 04 |
| `nagrania_gestow/` | Krotkie filmy per gest | 05 |
| `nagrania_testy/` | Filmy do testow porownawczych | 03 |
| `analiza/` | Obrazy analizy wzorcow | 01 |

## Szybki start

### 1. Konfiguracja srodowisk

```bat
setup_venv.bat
```

Skrypt stworzy folder `venv/` w kazdym etapie i zainstaluje wymagane pakiety.

### 2. Aktywacja srodowiska dla wybranego etapu

```powershell
.\etap_02_siec_mlp\venv\Scripts\activate.ps1
cd etap_02_siec_mlp
python trenowanie.py
```

## Opis etapow

### Etap 1 — Dopasowanie szablonow (`etap_01_dopasowanie_szablonow`)

Pierwsza proba: rozpoznawanie bez uczenia maszynowego. Program porownuje geometrie 21 punktow dłoni z kamery do przygotowanych wczesniej zdjec wzorcowych. Zbiera automatycznie klatki do folderu `do_nauki/` gdy pewnosc > 85%.

**Uruchomienie:**
```
python rozpoznawanie_rak.py
```

---

### Etap 2 — Siec neuronowa MLP (`etap_02_siec_mlp`)

Zebrane dane (zdjecia) sa analizowane przez MediaPipe, a wyekstrahowane cechy (43 liczby: 21 punktow 2D + kat) trafia do sieci MLP (Keras). Workflow:

1. `python robienie_zdjec.py` — zbierz zdjecia do `../lepsze_dane/`
2. `python trenowanie.py` — wytrenuj model, zapisze `model_gesty_punkty.keras`
3. `python pokazanie.py` — demo na zywo

---

### Etap 3 — XGBoost i porownanie (`etap_03_xgboost_porownanie`)

Eksperyment z modelem XGBoost zamiast sieci neuronowej, + wizualne porownanie obu modeli na zywo i na nagraniu.

1. `python xgb_trenowanie.py` — trenowanie XGBoost (potrzebne dane z etapu 2 w `../lepsze_dane/`)
2. `python wizualizacja_porownanie.py` — kamera z wynikami obu modeli obok siebie
3. `python nagranie_porownij.py` — przetworz plik `../nagrania_testy/alfabet.mp4`

**Uwaga:** Wymaga gotowych modeli z etapu 2 (`model_gesty_punkty.keras`, `etykiety_punkty.pkl`) skopiowanych do tego folderu.

---

### Etap 4 — Keras Tuner + MediaPipe Tasks API (`etap_04_keras_tuner`)

Przejscie na nowe MediaPipe Tasks API (3D landmarks + informacja o rece). Automatyczne wyszukiwanie najlepszej architektury sieci przez Bayesian Optimization (Keras Tuner). Trening nocny.

1. Upewnij sie ze dane sa w `../lepsze_dane/<LITERA>/` (podfoldery — uzyj `zmiana_do_folder.py` do konwersji z flat)
2. `odpal.bat` — uruchamia `skrypt_treningowy.py` i loguje czas do `raport_nocny.txt`
3. `python reprezentacja_trenowanego.py` — demo (kamera / wideo / zdjecie)
4. `python nowe_demo.py` — wersja PRO z obracajacym sie modelem 3D dłoni

---

### Etap 5 — Gesty dynamiczne LSTM (`etap_05_lstm_ruch`)

Rozpoznawanie gestow wymagajacych ruchu (np. litery J, Z w PJM). Model LSTM uczy sie sekwencji 30 klatek.

1. `python nagrywanie_filmow_do_danych.py` — nagraj krotkie filmy do `../nagrania_gestow/`
2. `python terning_na_filmie.py` — trenowanie LSTM, zapisze `model_gesty_ruchome.keras`
3. `python ruch_pokaz.py` — demo na zywo z historia ostatnich 3 gestow

---

## Wymagania systemowe

- Python 3.11 (64-bit)
- Kamera internetowa
- Windows (testowane), Linux/macOS (powinno dzialac)
- GPU opcjonalne (TensorFlow obsluguje CPU)
