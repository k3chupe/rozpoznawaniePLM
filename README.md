# Rozpoznawanie Gestów Dłoni (MediaPipe + OpenCV + TensorFlow)

Repozytorium zawiera pełny pipeline do budowy własnego rozpoznawania gestów dłoni:

1. przygotowanie wzorców gestów,
2. zebranie danych treningowych kamerą,
3. trening modelu,
4. test modelu na żywo.

## Co robi każdy skrypt

- `rozpoznwaanie_rak1.py`
    - wczytuje wzorce z folderu `alfabet/` (pliki `wzor_*.jpg`),
    - porównuje gest z kamery do wzorców,
    - automatycznie zapisuje zdjęcia do `do_nauki/` (gdy pewność > 85%).
- `trenowanie.py`
    - analizuje zdjęcia z `do_nauki/`,
    - trenuje model MLP,
    - zapisuje `model_gesty_punkty.keras` i `etykiety_punkty.pkl`.
- `pokazanie.py`
    - ładuje wytrenowany model,
    - pokazuje rozpoznaną literę i pewność na obrazie z kamery.

## Wymagania

- Python 3.11 (64-bit) zalecany
- Kamera internetowa
- Windows / Linux / macOS

## Instalacja

1. Sklonuj repozytorium:

```bash
git clone https://github.com/k3chupe/rozpoznawaniePLM.git
cd rozpoznawaniePLM
```

2. Utwórz i aktywuj środowisko wirtualne:

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

## Krok 1: Przygotuj wzorce w folderze alfabet

Do folderu `alfabet/` dodaj zdjęcia wzorcowe, po jednym na każdy gest/literę, np.:

- `wzor_A.jpg`
- `wzor_B.jpg`
- `wzor_C.jpg`

Ważne:

- nazwa musi zaczynać się od `wzor_` i kończyć na `.jpg`,
- litera w nazwie pliku to etykieta klasy,
- zdjęcie powinno zawierać wyraźnie widoczną dłoń (najlepiej jednolite tło i dobre światło).

## Krok 2: Zbieranie danych do uczenia

Uruchom:

```bash
python rozpoznwaanie_rak1.py
```

Co zobaczysz:

- podgląd kamery z punktami dłoni,
- bieżącą rozpoznaną literę `ZNAK`,
- procenty dopasowania dla każdej litery ze wzorców.

Jak działa zapis danych:

- jeżeli gest jest rozpoznany (nie `0`) i ma pewność > 85%,
- skrypt zapisze surową klatkę do `do_nauki/`,
- zapis następuje maksymalnie raz na 3 sekundy,
- nazwa pliku ma format np. `A_1712345678901.jpg`.

Praktyczna wskazówka:

- dla każdej litery zbierz minimum 30-50 zdjęć,
- zmieniaj lekko kąt dłoni, odległość i tło,
- im bardziej zróżnicowane próbki, tym lepsza generalizacja modelu.

## Krok 3: Trening własnego modelu

Gdy masz dane w `do_nauki/`, uruchom:

```bash
python trenowanie.py
```

Skrypt:

- wyciąga cechy z 21 punktów dłoni (43 wartości),
- dzieli dane na zbiór treningowy i testowy,
- trenuje sieć neuronową,
- nadpisuje pliki:
    - `model_gesty_punkty.keras`
    - `etykiety_punkty.pkl`

Na końcu treningu w konsoli zobaczysz metryki, m.in. `accuracy` i `val_accuracy`.

## Krok 4: Test skuteczności na żywo

Uruchom:

```bash
python pokazanie.py
```

W oknie kamery zobaczysz:

- przewidywaną literę,
- procent pewności (pokazywany, gdy pewność > 60%).

W ten sposób sprawdzisz, jak model radzi sobie z gestami nauczonymi na Twoich danych.

## Szybki workflow (skrót)

1. Dodaj `wzor_*.jpg` do `alfabet/`.
2. Uruchom `python rozpoznwaanie_rak1.py` i zbierz dane do `do_nauki/`.
3. Uruchom `python trenowanie.py`.
4. Uruchom `python pokazanie.py` i oceń skuteczność.

## Rozwiązywanie problemów

- Kamera się nie uruchamia:
    - sprawdź, czy inna aplikacja nie blokuje kamery,
    - zmień indeks kamery w kodzie (`cv2.VideoCapture(0)` -> `1`).
- Brak zapisu zdjęć do `do_nauki/`:
    - upewnij się, że masz poprawne wzorce w `alfabet/`,
    - sprawdź, czy na ekranie pojawia się litera inna niż `0`,
    - utrzymuj gest stabilnie przez kilka sekund.
- Słabe wyniki modelu:
    - dozbieraj więcej danych,
    - wyrównaj liczbę próbek między klasami,
    - zadbaj o różne warunki oświetlenia i tła.