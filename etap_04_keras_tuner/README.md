# Etap 4 — Keras Tuner + MediaPipe Tasks API

**Cel:** Automatyczne znajdowanie najlepszej architektury sieci przez Bayesian Optimization (Keras Tuner) oraz przejscie na nowe MediaPipe Tasks API z punktami 3D.

**Czego sie nauczylem:**
- MediaPipe Tasks API (`HandLandmarker`) vs stare Solutions API
- Cechy 3D (21 punktow x,y,z) + bit reki = 64 cechy
- Keras Tuner: Bayesian Optimization, `max_trials`, `project_name`
- BatchNormalization przed aktywacja
- ReduceLROnPlateau jako dodatkowy callback

## Wymagania wstepne

Dane w `../lepsze_dane/<LITERA>/` (podfoldery per klasa).
Jesli masz flat (`lepsze_dane/A_*.jpg`), uzyj najpierw `zmiana_do_folder.py`.

## Workflow

```powershell
.\venv\Scripts\Activate.ps1

# Opcja A: Jednorazowe uruchomienie
python skrypt_treningowy.py

# Opcja B: Trening nocny (loguje czas w raport_nocny.txt)
odpal.bat

# Demo po treningu
python reprezentacja_trenowanego.py
python nowe_demo.py
```

## Pliki wejsciowe / wyjsciowe

| Plik | Opis |
|---|---|
| `../lepsze_dane/<LITERA>/` | Zdjecia treningowe w podfolderach |
| `hand_landmarker.task` | Model MediaPipe (pobierany automatycznie) |
| `model_gesty_punkty_v2_nocny.keras` | Wytrenowany model |
| `etykiety_punkty_v2_nocny.pkl` | Etykiety klas |
| `moje_poszukiwania_nocc/` | Cache Keras Tuner (ignorowany przez git) |
| `raport_nocny.txt` | Logi czasu treningu nocnego |

## Architektura (szukana przez Tuner)

```
Input(64) -> Dense(128-1024) -> BN -> ReLU -> Dropout(0.1-0.6)
          -> Dense(64-512)  -> BN -> ReLU -> Dropout(0.1-0.6)
          -> [opcjonalnie Dense(32-256) -> BN -> ReLU -> Dropout]
          -> Dense(N_klas, softmax)
```

## Naprawione bugi (wzgledem oryginalnego kodu)

- Augmentacja: wczeniej `+=` mylnie sumowalo piksele zamiast dodawac osobne probki — teraz kazda rotacja to oddzielne wywolanie `analizuj_i_dodaj`
- `nowe_demo.py`: laduje teraz `etykiety_punkty_v2_nocny.pkl` (odpowiadajace modelowi v2_nocny)
- `reprezentacja_trenowanego.py`: laduje teraz `model_gesty_punkty_v2_nocny.keras`

## Wniosek

Keras Tuner znalazl lepsze architektury niz reczne strojenie, ale czas treningu bardzo dlugi. Wyniki zaskakujaco slabsze niz oczekiwano. Przejscie do etapu 5 — gesty dynamiczne.
