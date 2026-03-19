# Rozpoznawanie Gestów Dłoni (MediaPipe & OpenCV) 🖐️

Prosty, ale potężny program napisany w języku Python, który używa kamery internetowej oraz sieci neuronowej (MediaPipe) do wykrywania dłoni w czasie rzeczywistym. Program wczytuje zdjęcie ze wzorem gestu (`wzor_W.jpg`), oblicza jego matematyczny szkielet i porównuje z tym, co aktualnie widzi kamera.

Algorytm jest odporny na obrót dłoni oraz jej odległość od obiektywu!

## 🚀 Jak uruchomić gotowy program (bez programowania)?
Jeśli chcesz po prostu przetestować program, przejdź do zakładki **[Releases]** po prawej stronie i pobierz najnowszy plik `.zip`. 
Wypakuj go, upewnij się, że plik `.exe` oraz `wzor_w.jpg` są w tym samym folderze i uruchom aplikację.

## 💻 Jak uruchomić kod ze źródeł (dla programistów)?

### Wymagania:
- Python 3.11 lub 3.12 (wersja 64-bitowa)
- Kamera internetowa

### Instalacja:
1. Sklonuj to repozytorium:
   ```bash
    git clone https://github.com/k3chupe/rozpoznawaniePLM.git
    cd rozpoznawaniePLM

2. Stwórz i aktywuj wirtualne środowisko:
Bash

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

3. Zainstaluj wymagane biblioteki:
Bash

pip install -r requirements.txt


### Uruchamianie:

Upewnij się, że w folderze z projektem znajduje się zdjęcie wzor_w.jpg przedstawiające gest (skierowany wyraźnie do góry). Następnie wpisz:
Bash

python rozpoznwaanie_rak1.py