import os
import shutil

# ==========================================
# KONFIGURACJA
# ==========================================
NAZWA_FOLDERU = "lepsze_dane"

def posortuj_zdjecia():
    # 1. Sprawdzamy, czy folder w ogóle istnieje
    if not os.path.exists(NAZWA_FOLDERU):
        print(f"Błąd: Nie znaleziono folderu o nazwie '{NAZWA_FOLDERU}'.")
        return

    licznik = 0
    
    # 2. Przechodzimy przez wszystkie elementy w folderze
    print(f"Rozpoczynam sortowanie plików w folderze '{NAZWA_FOLDERU}'...")
    for plik in os.listdir(NAZWA_FOLDERU):
        sciezka_pliku = os.path.join(NAZWA_FOLDERU, plik)
        
        # 3. Interesują nas tylko pliki (ignorujemy podfoldery) z rozszerzeniem .jpg / .jpeg
        if os.path.isfile(sciezka_pliku) and plik.lower().endswith(('.jpg', '.jpeg', '.png')):
            
            # Pobieramy pierwszą literę nazwy pliku i zamieniamy na wielką
            pierwsza_litera = plik[0].upper()
            
            # Tworzymy ścieżkę do docelowego podfolderu
            sciezka_podfolderu = os.path.join(NAZWA_FOLDERU, pierwsza_litera)
            
            # 4. Jeśli podfolder (np. "A") nie istnieje, tworzymy go
            if not os.path.exists(sciezka_podfolderu):
                os.makedirs(sciezka_podfolderu)
                
            # Ścieżka, pod którą ma ostatecznie wylądować plik
            nowa_sciezka_pliku = os.path.join(sciezka_podfolderu, plik)
            
            # 5. Przenosimy plik
            shutil.move(sciezka_pliku, nowa_sciezka_pliku)
            licznik += 1
            print(f"Przeniesiono: {plik} -> {pierwsza_litera}/")

    print("\n" + "="*40)
    print(f"Gotowe! Pomyślnie posortowano {licznik} plików.")
    print("="*40)

if __name__ == "__main__":
    posortuj_zdjecia()