import cv2
import mediapipe as mp
import os
import time
import keyboard

# --- KONFIGURACJA ---
NAZWA_FOLDERU = "../nagrania_gestow"
PLIK_PLANU = "plan_nagrania.txt"
ROZDZIELCZOSC = (1920, 1080)
FPS = 30.0

if not os.path.exists(NAZWA_FOLDERU):
    os.makedirs(NAZWA_FOLDERU)

# 1. Tworzenie domyślnego pliku z planem (jeśli go nie ma)
if not os.path.exists(PLIK_PLANU):
    with open(PLIK_PLANU, "w") as f:
        f.write("H 10\nJ 10\nZ 15")
    print(f"Utworzono domyślny plik '{PLIK_PLANU}'. Możesz go wyedytować, by zmienić plan!")

# 2. Wczytywanie planu
harmonogram = []
with open(PLIK_PLANU, "r") as f:
    for linia in f:
        czesci = linia.strip().split()
        if len(czesci) == 2:
            litera = czesci[0].upper()
            ilosc = int(czesci[1])
            if ilosc > 0:
                harmonogram.append([litera, ilosc])

if not harmonogram:
    print("Twój harmonogram jest pusty lub źle sformatowany! Zamykam program.")
    exit()

# Zmienne śledzące aktualny postęp w harmonogramie
aktualny_krok = 0
aktualna_litera = harmonogram[aktualny_krok][0]
ilosc_docelowa = harmonogram[aktualny_krok][1]
nagran_zrobionych = 0

# Upewnienie się, że folder dla pierwszej litery z planu istnieje
os.makedirs(os.path.join(NAZWA_FOLDERU, aktualna_litera), exist_ok=True)

# --- INICJALIZACJA KAMERY I MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ROZDZIELCZOSC[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ROZDZIELCZOSC[1])

is_recording = False
out = None

print("\n--- KAMERA URUCHOMIONA ---")
print("PRZYTRZYMAJ [SPACJĘ], by nagrywać ruch. PUŚĆ, aby zapisać film.")
print("Wciśnij [Q] w oknie kamery, aby wyjść z programu.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Błąd odczytu z kamery.")
        break

    # Efekt lustra dla naturalnego podglądu
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # Rysowanie punktów MediaPipe (tylko na podglądzie, nie na nagraniu!)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    wyniki = hands.process(frame_rgb)
    if wyniki.multi_hand_landmarks:
        for hand_landmarks in wyniki.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- LOGIKA NAGRYWANIA (TRZYMANIE SPACJI) ---
    if keyboard.is_pressed('space'):
        if not is_recording:
            # START NAGRYWANIA
            is_recording = True
            sciezka_klasy = os.path.join(NAZWA_FOLDERU, aktualna_litera)
            nazwa_pliku = f"{aktualna_litera}_{int(time.time() * 1000)}.mp4"
            sciezka_zapisu = os.path.join(sciezka_klasy, nazwa_pliku)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(sciezka_zapisu, fourcc, FPS, ROZDZIELCZOSC)
            print(f"Nagrywanie: {nazwa_pliku}...")

        # Zapis czystej klatki do pliku
        out.write(frame)
        
        # Wyświetlanie UI o nagrywaniu
        cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1)
        cv2.putText(display_frame, "NAGRYWANIE RUCHU...", (80, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    
    else:
        if is_recording:
            # STOP NAGRYWANIA (Puszczono spację)
            is_recording = False
            out.release()
            nagran_zrobionych += 1
            print(f"[ZAPISANO] Wykonano {nagran_zrobionych}/{ilosc_docelowa} nagrań dla '{aktualna_litera}'.")

            # --- SPRAWDZANIE CZY SKOŃCZYLIŚMY AKTUALNĄ LITERĘ ---
            if nagran_zrobionych >= ilosc_docelowa:
                aktualny_krok += 1
                
                # Czy to był koniec całego planu?
                if aktualny_krok >= len(harmonogram):
                    print("\n=== GRATULACJE! CAŁY PLAN WYKONANY! ===")
                    cv2.rectangle(display_frame, (10, 10), (800, 150), (0, 255, 0), -1)
                    cv2.putText(display_frame, "PLAN ZAKONCZONY!", (30, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3)
                    cv2.imshow('Zbieranie Danych', cv2.resize(display_frame, (960, 540)))
                    cv2.waitKey(3000) # Pokaż ekran końcowy przez 3 sekundy
                    break
                else:
                    # Przejście do następnej litery
                    aktualna_litera = harmonogram[aktualny_krok][0]
                    ilosc_docelowa = harmonogram[aktualny_krok][1]
                    nagran_zrobionych = 0
                    
                    # Utworzenie nowego folderu dla kolejnej litery
                    os.makedirs(os.path.join(NAZWA_FOLDERU, aktualna_litera), exist_ok=True)
                    
                    print(f"\n---> ZMIANA ZADANIA! Teraz nagrywaj: {aktualna_litera} <---")

    # --- UI STATUSU NA EKRANIE ---
    tekst_info = f"Zadanie: {aktualna_litera} | Zrobiono: {nagran_zrobionych}/{ilosc_docelowa}"
    cv2.rectangle(display_frame, (10, 80), (700, 130), (0, 0, 0), -1)
    cv2.putText(display_frame, tekst_info, (20, 115), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

    # Wyświetlanie obrazu (zmniejszonego o połowę, żeby Full HD zmieściło się wygodnie na monitorze)
    cv2.imshow('Zbieranie Danych (Spacja = Nagrywaj)', cv2.resize(display_frame, (960, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Przerwano przez użytkownika (Klawisz Q).")
        break

# Sprzątanie
if is_recording and out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
hands.close()