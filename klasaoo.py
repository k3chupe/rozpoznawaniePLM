import cv2
import mediapipe as mp
import os
import time

# --- Konfiguracja ---
NAZWA_FOLDERU = "do_nauki"
# Tworzymy folder, jeśli nie istnieje
if not os.path.exists(NAZWA_FOLDERU):
    os.makedirs(NAZWA_FOLDERU)
    print(f"Utworzono folder: {NAZWA_FOLDERU}")

# --- Inicjalizacja MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# static_image_mode=False dla kamery na żywo
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# --- Uruchomienie kamery ---
cap = cv2.VideoCapture(0)

print("\n--- PROGRAM URUCHOMIONY ---")
print("Ustaw rękę w pozycji, którą chcesz zapisać (np. losowe ruchy dla klasy 0).")
print("Wciśnij [SPACJA], aby zrobić i zapisać zdjęcie.")
print("Wciśnij [Q], aby wyjść z programu.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Nie można odebrać klatki z kamery.")
        break

    # Odbicie lustrzane dla wygody użytkownika
    frame = cv2.flip(frame, 1)
    
    # Tworzymy kopię obrazu RGB dla MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    wyniki = hands.process(frame_rgb)

    # Obraz, który będziemy wyświetlać użytkownikowi (z rysunkami)
    display_frame = frame.copy()

    # Jeśli znaleziono dłoń, rysujemy szkielet
    if wyniki.multi_hand_landmarks:
        for hand_landmarks in wyniki.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Wyświetlamy obraz z nałożonym szkieletem
    cv2.imshow('Zbieranie Danych (Wcisnij SPACJE)', display_frame)

    # Reakcja na klawisze
    key = cv2.waitKey(1) & 0xFF
    
    # Wyjście z programu po wciśnięciu 'q'
    if key == ord('q'):
        break
        
    # Zapisanie zdjęcia po wciśnięciu 'spacji'
    elif key == 32: # 32 to kod klawisza 'spacja'
        # Generujemy unikalną nazwę pliku (0_ + timestamp)
        # int(time.time() * 1000) daje unikalny, długi numer milisekundowy
        unikalny_numer = int(time.time() * 1000)
        nazwa_pliku = f"R_{unikalny_numer}.jpg"
        sciezka_zapisu = os.path.join(NAZWA_FOLDERU, nazwa_pliku)
        
        # ZAPISUJEMY ORYGINALNĄ KLATKĘ (frame), a nie tę z narysowanym szkieletem!
        cv2.imwrite(sciezka_zapisu, frame)
        print(f"Zapisano zdjęcie: {sciezka_zapisu}")

# Sprzątanie
cap.release()
cv2.destroyAllWindows()
hands.close()