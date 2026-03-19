import cv2
import mediapipe as mp
import time
import math # Nowa biblioteka do obliczeń matematycznych
import sys  # Do awaryjnego zamykania programu

# -------------------------------------------------------------
# CZĘŚĆ 1: FUNKCJE DO MATEMATYCZNEGO OBLICZANIA GESTU
# -------------------------------------------------------------

def pobierz_szkielet_gestu(hand_landmarks):
    """Zamienia dłoń na wektor proporcji, całkowicie odporny na obrót i odległość (Punkt 9 zawsze w 0,1)"""
    punkty = hand_landmarks.landmark
    nadgarstek = punkty[0]
    srodek_dloni = punkty[9]
    
    # 1. Obliczamy wektor kierunkowy od nadgarstka do środka dłoni (nasza nowa oś Y)
    dx = srodek_dloni.x - nadgarstek.x
    dy = srodek_dloni.y - nadgarstek.y
    
    # 2. Obliczamy kwadrat odległości (potrzebny do odpowiedniego skalowania)
    d_sq = dx**2 + dy**2
    if d_sq == 0: 
        d_sq = 0.0001 # Zabezpieczenie przed błędem dzielenia przez zero
        
    wektor = []
    
    # 3. Przekształcamy każdy z 21 punktów do nowego układu współrzędnych
    for punkt in punkty:
        # Krok A (Translacja): Przesuwamy nadgarstek do punktu (0,0)
        px = punkt.x - nadgarstek.x
        py = punkt.y - nadgarstek.y
        
        # Krok B (Rotacja i Skalowanie): Magia matematyki wektorów!
        # Używamy iloczynu wektorowego i skalarnego, aby tak obrócić i przeskalować siatkę, 
        # żeby punkt [9] wylądował dokładnie w (0, 1).
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
    
    # --- NOWE: Obliczanie pewności (0-100%) ---
    # Uznajemy, że średnia różnica na poziomie 0.4 oznacza zupełnie inny gest (0% podobieństwa)
    max_roznica = 0.4 
    pewnosc = 100 - ((srednia_roznica / max_roznica) * 100)
    
    # Nie chcemy ujemnych procentów, jeśli dłoń jest wybitnie inna
    if pewnosc < 0: 
        pewnosc = 0
        
    czy_pasuje = srednia_roznica < tolerancja
    
    return czy_pasuje, pewnosc

# -------------------------------------------------------------
# CZĘŚĆ 2: KONFIGURACJA MEDIAPIPE I WCZYTANIE WZORU
# -------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Model do obrazka statycznego (wzoru) i do kamery (wideo)
hands_wzor = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
hands_kamera = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

print("Wczytuję obraz ze wzorem gestu...")
obraz_wzoru = cv2.imread("wzor_w.jpg")

if obraz_wzoru is None:
    print("BŁĄD: Nie znaleziono pliku 'wzor.jpg'! Upewnij się, że jest w tym samym folderze.")
    sys.exit()

# Przetworzenie obrazka ze wzorem
obraz_wzoru_rgb = cv2.cvtColor(obraz_wzoru, cv2.COLOR_BGR2RGB)
wynik_wzoru = hands_wzor.process(obraz_wzoru_rgb)

if not wynik_wzoru.multi_hand_landmarks:
    print("BŁĄD: Sieć nie wykryła dłoni na zdjęciu wzorcowym! Zrób lepsze zdjęcie.")
    sys.exit()

# Zapisujemy nasz wzorzec matematyczny na podstawie zdjęcia
nasz_wzor_gestu = pobierz_szkielet_gestu(wynik_wzoru.multi_hand_landmarks[0])
print("Wzór gestu zapisany pomyślnie!")

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
    
    czy_wykryto_gest = False 
    najlepsza_pewnosc = 0 # Nowa zmienna do przechowywania wyniku

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            szkielet_na_zywo = pobierz_szkielet_gestu(hand_landmarks)
            
            # Funkcja zwraca teraz dwie wartości
            czy_pasuje, pewnosc = porownaj_gesty(nasz_wzor_gestu, szkielet_na_zywo, tolerancja=0.20)
            
            if czy_pasuje:
                czy_wykryto_gest = True
                
            # Zapisujemy wynik dłoni, która jest najbardziej podobna do wzoru
            if pewnosc > najlepsza_pewnosc:
                najlepsza_pewnosc = pewnosc

    # --- ZAKTUALIZOWANE WYŚWIETLANIE TEKSTU ---
    if czy_wykryto_gest:
        tekst = f'GEST: TAK ({int(najlepsza_pewnosc)}%)'
        cv2.putText(image, tekst, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        tekst = f'GEST: NIE ({int(najlepsza_pewnosc)}%)'
        cv2.putText(image, tekst, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Obliczanie FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Pokazywanie okna
    cv2.imshow('Wykrywanie Dloni - MediaPipe', image)

    # Zamykanie klawiszem 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
        
    # Zamykanie iksem 'X'
    if cv2.getWindowProperty('Wykrywanie Dloni - MediaPipe', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()