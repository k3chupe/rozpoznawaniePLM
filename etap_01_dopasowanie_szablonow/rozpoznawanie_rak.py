import cv2
import mediapipe as mp
import time
import sys
import os

# -------------------------------------------------------------
# CZĘŚĆ 1: FUNKCJE MATEMATYCZNE (Mózg: 21 punktów)
# -------------------------------------------------------------

def pobierz_szkielet_gestu(hand_landmarks):
    """Zwraca precyzyjny wektor ze wszystkich 21 punktów (100% skuteczności)"""
    punkty = hand_landmarks.landmark
    nadgarstek = punkty[0]
    srodek_dloni = punkty[9] 
    
    dx = srodek_dloni.x - nadgarstek.x
    dy = srodek_dloni.y - nadgarstek.y
    d_sq = dx**2 + dy**2
    if d_sq == 0: d_sq = 0.0001
        
    wektor = []
    # Lecimy przez wszystkie punkty, dłoń jest analizowana perfekcyjnie
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
    return srednia_roznica < tolerancja, pewnosc

# -------------------------------------------------------------
# CZĘŚĆ 2: KONFIGURACJA I FOLDERY
# -------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # Wracamy do oficjalnego rysowania!

hands_wzor = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
hands_kamera = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

nazwa_folderu = '../alfabet'
folder_analizy = '../analiza'
folder_do_nauki = '../do_nauki'

if not os.path.exists(nazwa_folderu):
    print(f"BŁĄD: Nie znaleziono folderu '{nazwa_folderu}'!")
    sys.exit()

os.makedirs(folder_analizy, exist_ok=True)
os.makedirs(folder_do_nauki, exist_ok=True)

konfiguracja_gestow = {}
for plik in os.listdir(nazwa_folderu):
    if plik.startswith('wzor_') and plik.endswith('.jpg'):
        litera = plik.replace('wzor_', '').replace('.jpg', '').upper()
        konfiguracja_gestow[litera] = os.path.join(nazwa_folderu, plik)

if not konfiguracja_gestow:
    print(f"BŁĄD: Folder '{nazwa_folderu}' jest pusty.")
    sys.exit()

baza_wiedzy = {}
print("Trwa wczytywanie i analiza obrazów wzorcowych...")

for litera, sciezka in konfiguracja_gestow.items():
    obraz = cv2.imread(sciezka)
    if obraz is None: continue
        
    obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
    wynik = hands_wzor.process(obraz_rgb)
    
    if wynik.multi_hand_landmarks:
        reka_na_zdjeciu = wynik.multi_handedness[0].classification[0].label
        szkielet = pobierz_szkielet_gestu(wynik.multi_hand_landmarks[0])
        baza_wiedzy[litera] = szkielet#{'szkielet': szkielet, 'reka': reka_na_zdjeciu}
        
        # Wracamy do oryginalnego, pełnego rysowania 21 punktów z MediaPipe na zdjęciach wzorcowych
        kolor_wzoru = (0, 255, 0) if reka_na_zdjeciu == 'Right' else (255, 0, 0)
        mp_drawing.draw_landmarks(
            obraz, wynik.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=kolor_wzoru, thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=kolor_wzoru, thickness=2)
        )
        cv2.imwrite(os.path.join(folder_analizy, f"analiza_{litera}.jpg"), obraz)
        print(f"  -> Znak '{litera}' ({reka_na_zdjeciu}) wczytany.")

# -------------------------------------------------------------
# CZĘŚĆ 3: GŁÓWNA PĘTLA KAMERY I ZBIERANIE DANYCH
# -------------------------------------------------------------

cap = cv2.VideoCapture(0)
pTime = 0
ostatni_zapis = 0

print("Kamera uruchomiona. Zbieranie danych do ML aktywne!")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1) 
    surowy_obraz = image.copy() # Zapisujemy czystą klatkę, bez napisów
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_kamera.process(image_rgb)
    
    glowna_litera = '0'
    najlepsza_pewnosc_ogolna = 0
    procenty_na_ekranie = {litera: 0 for litera in konfiguracja_gestow}

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            etykieta_z_kamery = handedness.classification[0].label
            fizyczna_reka = 'Right' if etykieta_z_kamery == 'Left' else 'Left'
            kolor_rysowania = (255, 150, 200) if fizyczna_reka == 'Right' else (255, 255, 255)
            
            # Wracamy do oryginalnego, pełnego rysowania 21 punktów na żywo z kamery
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=kolor_rysowania, thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=kolor_rysowania, thickness=2)
            )
            
            szkielet_na_zywo = pobierz_szkielet_gestu(hand_landmarks)
            
            for litera, szkielet_wzoru in baza_wiedzy.items():
                czy_pasuje, pewnosc = porownaj_gesty(szkielet_wzoru, szkielet_na_zywo, tolerancja=0.20)
                
                # Zapisujemy najwyższy procent dla danej litery (w razie jakby były dwie dłonie na kamerze)
                if pewnosc > procenty_na_ekranie[litera]:
                    procenty_na_ekranie[litera] = pewnosc
                    
                # Jeśli ten gest pasuje najlepiej ze wszystkich, ustawiamy go jako główną literę
                if czy_pasuje and pewnosc > najlepsza_pewnosc_ogolna:
                    najlepsza_pewnosc_ogolna = pewnosc
                    glowna_litera = litera

    # ---------------------------------------------------------
    # ZAPISYWANIE DANYCH DO MACHINE LEARNINGU (>85%)
    # ---------------------------------------------------------
    if glowna_litera != '0' and najlepsza_pewnosc_ogolna > 85:
        aktualny_czas = time.time()
        
        if aktualny_czas - ostatni_zapis > 3:
            nazwa_pliku = f"{glowna_litera}_{int(aktualny_czas * 1000)}.jpg"
            sciezka_zapisu = os.path.join(folder_do_nauki, nazwa_pliku)
            cv2.imwrite(sciezka_zapisu, surowy_obraz)
            ostatni_zapis = aktualny_czas
            cv2.circle(image, (50, 150), 10, (0, 0, 255), -1)

    # --- WYŚWIETLANIE WYNIKÓW ---
    kolor_litery = (0, 255, 0) if glowna_litera != '0' else (0, 0, 255)
    cv2.putText(image, f'ZNAK: {glowna_litera}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, kolor_litery, 3)

    wysokosc_tekstu = 120
    for litera, procent in procenty_na_ekranie.items():
        cv2.putText(image, f'{litera}: {int(procent)}%', (10, wysokosc_tekstu), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        wysokosc_tekstu += 21 # Zgodnie z wytycznymi

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Wykrywanie Dloni - MediaPipe', image)

    if cv2.waitKey(5) & 0xFF == ord('q'): break
    if cv2.getWindowProperty('Wykrywanie Dloni - MediaPipe', cv2.WND_PROP_VISIBLE) < 1: break

cap.release()
cv2.destroyAllWindows()