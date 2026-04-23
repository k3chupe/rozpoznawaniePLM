import cv2
import mediapipe as mp
import os
import time

# --- Konfiguracja ---
NAZWA_FOLDERU = "../lepsze_dane"
PLIK_PLANU = "plan_zbierania.txt"

if not os.path.exists(NAZWA_FOLDERU):
    os.makedirs(NAZWA_FOLDERU)
    print(f"Utworzono folder: {NAZWA_FOLDERU}")

# Tworzenie przykładowego pliku z planem, jeśli nie istnieje
if not os.path.exists(PLIK_PLANU):
    with open(PLIK_PLANU, "w") as f:
        f.write("A 10\nB 12\nD 9\n0 15")
    print(f"Utworzono domyślny plik '{PLIK_PLANU}'. Możesz go edytować, aby zmienić plan.")

# Wczytywanie planu z pliku
harmonogram = []
with open(PLIK_PLANU, "r") as f:
    for linia in f:
        czesci = linia.strip().split()
        if len(czesci) == 2:
            litera = czesci[0]
            ilosc = int(czesci[1])
            if ilosc > 0:
                harmonogram.append([litera, ilosc])

if not harmonogram:
    print("Twój harmonogram jest pusty lub źle sformatowany! Zamykam program.")
    exit()

# Zmienne śledzące aktualny postęp
aktualny_krok = 0
aktualna_litera = harmonogram[aktualny_krok][0]
pozostalo_zdjec = harmonogram[aktualny_krok][1]

# --- Inicjalizacja MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# --- Uruchomienie kamery ---
cap = cv2.VideoCapture(0)

print("\n--- PROGRAM URUCHOMIONY ---")
print("Wciśnij [SPACJA], aby zapisać zdjęcie.")
print("Wciśnij [Q], aby wyjść z programu.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Nie można odebrać klatki z kamery.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    wyniki = hands.process(frame_rgb)

    display_frame = frame.copy()

    if wyniki.multi_hand_landmarks:
        for hand_landmarks in wyniki.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # --- Wyświetlanie aktualnego zadania na ekranie ---
    tekst_info = f"Pokaz: {aktualna_litera} | Zostalo: {pozostalo_zdjec}"
    # Czarne tło pod tekstem dla czytelności
    cv2.rectangle(display_frame, (10, 10), (450, 60), (0, 0, 0), -1)
    # Wyświetlanie tekstu
    cv2.putText(display_frame, tekst_info, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Zbieranie Danych (Wcisnij SPACJE)', display_frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        
    elif key == 32: # Klawisz 'spacja'
        # Zapisujemy zdjęcie tylko, jeśli są jeszcze jakieś w kolejce
        if pozostalo_zdjec > 0:
            unikalny_numer = int(time.time() * 1000)
            # Używamy aktualnej litery z harmonogramu
            nazwa_pliku = f"{aktualna_litera}_{unikalny_numer}.jpg"
            sciezka_zapisu = os.path.join(NAZWA_FOLDERU, nazwa_pliku)
            
            cv2.imwrite(sciezka_zapisu, frame)
            print(f"[{aktualna_litera}] Zapisano zdjęcie: {nazwa_pliku}. Pozostało: {pozostalo_zdjec - 1}")
            
            pozostalo_zdjec -= 1

            # Przejście do następnej litery, jeśli skończyliśmy obecną
            if pozostalo_zdjec == 0:
                aktualny_krok += 1
                if aktualny_krok < len(harmonogram):
                    aktualna_litera = harmonogram[aktualny_krok][0]
                    pozostalo_zdjec = harmonogram[aktualny_krok][1]
                    print(f"\n---> ZMIANA ZADANIA! Teraz pokazuj: {aktualna_litera} <---")
                else:
                    print("\n=== GRATULACJE! Wszystkie zadania wykonane! ===")
                    # Ekran informujący o końcu
                    cv2.rectangle(display_frame, (10, 10), (450, 60), (0, 255, 0), -1)
                    cv2.putText(display_frame, "ZAKONCZONO!", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
                    cv2.imshow('Zbieranie Danych (Wcisnij SPACJE)', display_frame)
                    cv2.waitKey(2000) # Pokaż ekran końcowy przez 2 sekundy
                    break

cap.release()
cv2.destroyAllWindows()
hands.close()