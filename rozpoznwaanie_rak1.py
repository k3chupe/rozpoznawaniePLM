import cv2
import mediapipe as mp
import time
import sys
import os

# -------------------------------------------------------------
# CZĘŚĆ 1: FUNKCJE MATEMATYCZNE (Odporne na obrót)
# -------------------------------------------------------------

def pobierz_szkielet_gestu(hand_landmarks):
    punkty = hand_landmarks.landmark
    nadgarstek = punkty[0]
    srodek_dloni = punkty[9]
    
    dx = srodek_dloni.x - nadgarstek.x
    dy = srodek_dloni.y - nadgarstek.y
    d_sq = dx**2 + dy**2
    if d_sq == 0: d_sq = 0.0001
        
    wektor = []
    for punkt in punkty:
        px = punkt.x - nadgarstek.x
        py = punkt.y - nadgarstek.y
        nowy_x = (px * dy - py * dx) / d_sq
        nowy_y = (px * dx + py * dy) / d_sq
        wektor.append(nowy_x)
        wektor.append(nowy_y)
        
    return wektor

def porownaj_gesty(szkielet_wzoru, szkielet_kamery, tolerancja=0.20):
    suma_roznic = 0
    for w1, w2 in zip(szkielet_wzoru, szkielet_kamery):
        suma_roznic += abs(w1 - w2)
        
    srednia_roznica = suma_roznic / len(szkielet_wzoru)
    max_roznica = 0.4 
    pewnosc = 100 - ((srednia_roznica / max_roznica) * 100)
    
    if pewnosc < 0: pewnosc = 0
    czy_pasuje = srednia_roznica < tolerancja
    
    return czy_pasuje, pewnosc

# -------------------------------------------------------------
# CZĘŚĆ 2: KONFIGURACJA I WCZYTYWANIE WZORÓW
# -------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_wzor = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
hands_kamera = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# --- NOWOŚĆ: AUTOMATYCZNE SKANOWANIE FOLDERU ---
nazwa_folderu = 'alfabet'
konfiguracja_gestow = {}

# 1. Sprawdzamy, czy folder w ogóle istnieje, żeby uniknąć głupiego błędu
if not os.path.exists(nazwa_folderu):
    print(f"BŁĄD: Nie znaleziono folderu '{nazwa_folderu}'! Stwórz go obok pliku .py i wrzuć zdjęcia.")
    sys.exit()

# 2. Przeglądamy wszystkie pliki w tym folderze
for plik in os.listdir(nazwa_folderu):
    # Interesują nas tylko pliki zaczynające się od 'wzor_' i kończące na '.jpg'
    if plik.startswith('wzor_') and plik.endswith('.jpg'):
        
        # 3. Wyciągamy samą literę z nazwy pliku
        # Zamieniamy 'wzor_' na puste nic, '.jpg' na puste nic, a resztę zamieniamy na WIELKĄ literę
        litera = plik.replace('wzor_', '').replace('.jpg', '').upper()
        
        # 4. Łączymy nazwę folderu z nazwą pliku w bezpieczną ścieżkę (np. alfabet\wzor_a.jpg)
        pelna_sciezka = os.path.join(nazwa_folderu, plik)
        
        # 5. Zapisujemy w naszym słowniku!
        konfiguracja_gestow[litera] = pelna_sciezka

# Zabezpieczenie na wypadek pustego folderu
if not konfiguracja_gestow:
    print(f"BŁĄD: Folder '{nazwa_folderu}' jest pusty lub pliki mają złe nazwy.")
    print("Upewnij się, że nazywają się np. 'wzor_a.jpg', 'wzor_b.jpg'.")
    sys.exit()

print(f"Automatycznie znalazłem {len(konfiguracja_gestow)} gestów do wczytania: {list(konfiguracja_gestow.keys())}")

# Tutaj będziemy trzymać gotowe, matematyczne szkielety z obrazków
wzorce_szkieletow = {}

print("Trwa wczytywanie i analiza obrazów wzorcowych...")
for litera, nazwa_pliku in konfiguracja_gestow.items():
    obraz = cv2.imread(nazwa_pliku)
    if obraz is None:
        print(f"BŁĄD: Nie znaleziono pliku '{nazwa_pliku}'! Dodaj go do folderu.")
        sys.exit()
        
    obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
    wynik = hands_wzor.process(obraz_rgb)
    
    if not wynik.multi_hand_landmarks:
        print(f"BŁĄD: Nie wykryto dłoni na zdjęciu '{nazwa_pliku}'. Zrób wyraźniejsze zdjęcie.")
        sys.exit()
        
    # Zapisujemy wyliczony szkielet pod konkretną literą (np. pod 'A')
    szkielet = pobierz_szkielet_gestu(wynik.multi_hand_landmarks[0])
    wzorce_szkieletow[litera] = szkielet
    print(f"  -> Znak '{litera}' wczytany pomyślnie z pliku {nazwa_pliku}")

# -------------------------------------------------------------
# CZĘŚĆ 3: GŁÓWNA PĘTLA KAMERY
# -------------------------------------------------------------

cap = cv2.VideoCapture(0)
pTime = 0

print("Kamera uruchomiona. Pokaż dłoń! Wciśnij 'q', aby zamknąć okno.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_kamera.process(image_rgb)
    
    # Zmienne przechowujące najlepsze wyniki z obecnej klatki
    glowna_litera = '0'
    najlepsza_pewnosc_ogolna = 0
    
    # Dynamiczny słownik: sam tworzy sobie zera dla każdej litery z konfiguracji!
    procenty_na_ekranie = {litera: 0 for litera in konfiguracja_gestow}

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # --- RYSOWANIE DŁONI (Z KOLORAMI L/P) ---
            etykieta = handedness.classification[0].label
            kolor = (255, 255, 255) # Biały
            if etykieta == 'Left':  # Na lustrzanym odbiciu to Twoja prawa ręka
                kolor = (255, 150, 200) # Jasny fioletowy
                
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=kolor, thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=kolor, thickness=2)
            )
            
            # --- ROZPOZNAWANIE GESTU ---
            szkielet_na_zywo = pobierz_szkielet_gestu(hand_landmarks)
            
            # Porównujemy dłoń ze wszystkimi trzema wzorcami w pętli
            for litera, szkielet_wzoru in wzorce_szkieletow.items():
                czy_pasuje, pewnosc = porownaj_gesty(szkielet_wzoru, szkielet_na_zywo, tolerancja=0.20)
                
                # Zapisujemy najwyższy procent dla danej litery (w razie jakby były dwie dłonie na kamerze)
                if pewnosc > procenty_na_ekranie[litera]:
                    procenty_na_ekranie[litera] = pewnosc
                    
                # Jeśli ten gest pasuje najlepiej ze wszystkich, ustawiamy go jako główną literę
                if czy_pasuje and pewnosc > najlepsza_pewnosc_ogolna:
                    najlepsza_pewnosc_ogolna = pewnosc
                    glowna_litera = litera

    # --- WYŚWIETLANIE TEKSTÓW NA EKRANIE ---
    
    # 1. Główny, duży znak (Rozpoznana litera lub 0)
    kolor_litery = (0, 255, 0) if glowna_litera != '0' else (0, 0, 255)
    cv2.putText(image, f'ZNAK: {glowna_litera}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, kolor_litery, 3)

    # 2. Trzy mniejsze napisy z procentami podobieństwa
    wysokosc_tekstu = 120
    for litera, procent in procenty_na_ekranie.items():
        # Kolor żółty dla małych napisów
        cv2.putText(image, f'{litera}: {int(procent)}%', (10, wysokosc_tekstu), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        wysokosc_tekstu += 20 # Przesuwamy każdy kolejny tekst trochę niżej

    # Obliczanie FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Wykrywanie Dloni - MediaPipe', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Wykrywanie Dloni - MediaPipe', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()