import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras.models import load_model
import pickle
import time

# ==========================================
# 1. WCZYTANIE MODELU I ETYKIET
# ==========================================
print("Wczytywanie wytrenowanego modelu...")
try:
    model = load_model("model_gesty_punkty_v2.keras")
    with open("etykiety_punkty_v2.pkl", "rb") as f:
        lb = pickle.load(f)
except Exception as e:
    print(f"Błąd ładowania modelu! Upewnij się, że skrypt treningowy zakończył pracę. Błąd: {e}")
    exit()

# ==========================================
# 2. FUNKCJA POMOCNICZA (MUSI BYĆ IDENTYCZNA JAK W TRENINGU)
# ==========================================
def unifikuj_punkty(landmarks, handedness_category):
    punkty = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    nadgarstek = punkty[0]
    punkty_przesuniete = punkty - nadgarstek
    odleglosci = np.linalg.norm(punkty_przesuniete, axis=1)
    max_odleglosc = np.max(odleglosci)
    if max_odleglosc == 0: max_odleglosc = 1.0
    punkty_znormalizowane = punkty_przesuniete / max_odleglosc
    cechy = punkty_znormalizowane.flatten().tolist()
    
    etykieta_reki = handedness_category.category_name
    cechy.append(1.0 if etykieta_reki == 'Right' else 0.0)
    return np.array(cechy)

# ==========================================
# 3. GŁÓWNA FUNKCJA ROZPOZNAJĄCA
# ==========================================
def uruchom_detekcje(zrodlo="kamera", sciezka=None):
    """
    zrodlo: "kamera", "wideo", "zdjecie"
    sciezka: ścieżka do pliku (jeśli zrodlo to wideo lub zdjecie)
    """
    
    # Wybór trybu MediaPipe w zależności od źródła
    if zrodlo == "zdjecie":
        running_mode = vision.RunningMode.IMAGE
    else:
        # Zarówno wideo jak i kamera traktujemy jako strumień klatek w czasie
        running_mode = vision.RunningMode.VIDEO

    # Konfiguracja MediaPipe
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        
        # --- TRYB: ZDJĘCIE ---
        if zrodlo == "zdjecie":
            obraz = cv2.imread(sciezka)
            if obraz is None:
                print("Nie można wczytać zdjęcia.")
                return
            obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=obraz_rgb)
            
            wynik = detector.detect(mp_image)
            
            if wynik.hand_landmarks:
                cechy = unifikuj_punkty(wynik.hand_landmarks[0], wynik.handedness[0][0])
                cechy_wejscie = np.expand_dims(cechy, axis=0) # Dodanie wymiaru batch (1, 64)
                
                predykcja = model.predict(cechy_wejscie, verbose=0)
                indeks_klasy = np.argmax(predykcja)
                pewnosc = predykcja[0][indeks_klasy] * 100
                nazwa_gestu = lb.classes_[indeks_klasy]
                
                # Rysowanie na obrazku
                cv2.putText(obraz, f"Gest: {nazwa_gestu} ({pewnosc:.1f}%)", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(obraz, "Nie wykryto dloni", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Wynik", obraz)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # --- TRYB: KAMERA LUB WIDEO ---
        else:
            if zrodlo == "kamera":
                cap = cv2.VideoCapture(0) # 0 to domyślna kamera
            else:
                cap = cv2.VideoCapture(sciezka)
                
            if not cap.isOpened():
                print("Nie można otworzyć źródła wideo.")
                return

            print("Naciśnij 'q' aby zakończyć...")
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # Koniec wideo lub błąd kamery
                
                # Odbicie lustrzane dla wygody użytkownika (tylko z kamery)
                if zrodlo == "kamera":
                    frame = cv2.flip(frame, 1)

                obraz_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=obraz_rgb)
                
                # W trybie wideo musimy podać znacznik czasu (w milisekundach)
                timestamp_ms = int((time.time() - start_time) * 1000)
                
                wynik = detector.detect_for_video(mp_image, timestamp_ms)
                
                if wynik.hand_landmarks:
                    cechy = unifikuj_punkty(wynik.hand_landmarks[0], wynik.handedness[0][0])
                    cechy_wejscie = np.expand_dims(cechy, axis=0)
                    
                    predykcja = model.predict(cechy_wejscie, verbose=0)
                    indeks_klasy = np.argmax(predykcja)
                    pewnosc = predykcja[0][indeks_klasy] * 100
                    nazwa_gestu = lb.classes_[indeks_klasy]
                    
                    # Wypisz gest tylko jeśli model jest w miarę pewny
                    if pewnosc > 70.0:
                        tekst = f"Gest: {nazwa_gestu} ({pewnosc:.1f}%)"
                        kolor = (0, 255, 0) # Zielony
                    else:
                        tekst = f"Gest: Niezdecydowany..."
                        kolor = (0, 165, 255) # Pomarańczowy
                        
                    cv2.putText(frame, tekst, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, kolor, 2)
                    
                    # Opcjonalnie: Rysowanie kropki na nadgarstku dla bajeru wizualnego
                    nadgarstek = wynik.hand_landmarks[0][0]
                    h, w, c = frame.shape
                    cx, cy = int(nadgarstek.x * w), int(nadgarstek.y * h)
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

                cv2.imshow("Detekcja Gestow", frame)
                
                # Wyjście klawiszem 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()

# ==========================================
# 4. STEROWANIE - ODKOMENTUJ TO CO CHCESZ PRZETESTOWAĆ
# ==========================================

# Opcja 1: Rozpoznawanie na żywo z kamerki komputerowej
uruchom_detekcje(zrodlo="kamera")

# Opcja 2: Rozpoznawanie na zapisanym filmie
# uruchom_detekcje(zrodlo="wideo", sciezka="testowy_film.mp4")

# Opcja 3: Rozpoznawanie na pojedynczym zdjęciu
# uruchom_detekcje(zrodlo="zdjecie", sciezka="testowe_zdjecie.jpg")