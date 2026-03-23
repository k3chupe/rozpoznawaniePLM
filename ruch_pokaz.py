import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import time
from collections import deque # <--- Do tworzenia buforów (kolejek)

KAMERA_SZEROKOSC = 1280 # Zmniejszyłem trochę dla płynności, możesz dać 1920
KAMERA_WYSOKOSC = 720
MAX_KLATEK = 30 # Tyle klatek "pamięta" sieć LSTM

# Ładowanie MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("Ładowanie modelu LSTM...")
model = tf.keras.models.load_model("model_gesty_ruchome.keras")
with open("etykiety_ruch.pkl", "rb") as f:
    lb = pickle.load(f)

# --- KLUCZOWA ZMIANA: Ekstrakcja 64 cech (X, Y, Z + kąt) ---
def unifikuj_punkty(landmarks):
    punkty = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]) # Doszło Z
    nadgarstek = punkty[0]
    punkty_przesuniete = punkty - nadgarstek
    
    odleglosci = np.linalg.norm(punkty_przesuniete, axis=1)
    indeks_najdalszego = np.argmax(odleglosci)
    max_odleglosc = odleglosci[indeks_najdalszego]
    if max_odleglosc == 0: max_odleglosc = 1.0
        
    punkty_znormalizowane = punkty_przesuniete / max_odleglosc
    
    najdalszy_punkt = punkty_przesuniete[indeks_najdalszego]
    kat = math.atan2(najdalszy_punkt[1], najdalszy_punkt[0]) / math.pi
    
    cechy = punkty_znormalizowane.flatten().tolist() # Daje 63 elementy
    cechy.append(kat) # Daje 64. element
    return np.array(cechy)

def formatuj_nazwe(litera):
    if str(litera) == '0':
        return "Brak gestu"
    return str(litera)

# --- INICJALIZACJA BUFORÓW ---
# Kolejka, która automatycznie wyrzuca najstarszy element, gdy przekroczy 30
sekwencja_klatek = deque(maxlen=MAX_KLATEK) 
# Kolejka trzymająca 3 ostatnie rozpoznane gesty
historia_gestow = deque(maxlen=3) 

kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, KAMERA_SZEROKOSC)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, KAMERA_WYSOKOSC)
print("Kamera uruchomiona. Wciśnij 'q', aby wyjść.")

pTime = 0

while True:
    ret, ramka = kamera.read()
    if not ret: break
        
    ramka = cv2.flip(ramka, 1)
    ramka_rgb = cv2.cvtColor(ramka, cv2.COLOR_BGR2RGB)
    
    wynik = hands.process(ramka_rgb)
    
    if wynik.multi_hand_landmarks:
        for hand_landmarks in wynik.multi_hand_landmarks:
            mp_drawing.draw_landmarks(ramka, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cechy = unifikuj_punkty(hand_landmarks)
            sekwencja_klatek.append(cechy) # Dodajemy nową klatkę do "pamięci"
            
            # Sieć ocenia gest TYLKO wtedy, gdy uzbieramy pełne 30 klatek ruchu
            if len(sekwencja_klatek) == MAX_KLATEK:
                # Zmieniamy kształt z (30, 64) na (1, 30, 64) żeby weszło do sieci
                cechy_dla_modelu = np.expand_dims(np.array(sekwencja_klatek), axis=0)
                
                przewidywania = model(cechy_dla_modelu, training=False).numpy()[0]
                glowny_indeks = np.argmax(przewidywania)
                glowne_prawd = przewidywania[glowny_indeks]
                glowna_litera = formatuj_nazwe(lb.classes_[glowny_indeks])
                
                # Jeśli sieć jest bardzo pewna swojego wyniku
                if glowne_prawd > 0.85: 
                    tekst = f"Wykryto: {glowna_litera} ({glowne_prawd * 100:.1f}%)"
                    cv2.rectangle(ramka, (40, 45), (450, 95), (0,0,0), -1)
                    cv2.putText(ramka, tekst, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # LOGIKA HISTORII: Dodaj do historii tylko jeśli to nowy gest,
                    # żeby uniknąć spamowania ekranu np. 3x literą "H"
                    if len(historia_gestow) == 0 or historia_gestow[-1] != glowna_litera:
                        historia_gestow.append(glowna_litera)
                        # Po pomyślnym wykryciu czyścimy bufor ruchu!
                        # Dzięki temu algorytm musi zaczekać na kolejny, pełny nowy ruch, by coś zgadnąć
                        sekwencja_klatek.clear() 

    # --- RYSOWANIE PANELU Z HISTORIĄ (Prawy górny róg) ---
    prawy_margines = KAMERA_SZEROKOSC - 350
    cv2.rectangle(ramka, (prawy_margines - 20, 20), (KAMERA_SZEROKOSC - 20, 160), (30, 30, 30), -1)
    cv2.putText(ramka, "Ostatnie 3 gesty:", (prawy_margines, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # Rysujemy elementy z historii od najnowszego do najstarszego
    y_historia = 85
    for i, gest_z_historii in enumerate(reversed(historia_gestow)):
        kolor = (0, 255, 255) if i == 0 else (150, 150, 150) # Najnowszy jest żółty, reszta szara
        cv2.putText(ramka, f"- {gest_z_historii}", (prawy_margines + 20, y_historia), cv2.FONT_HERSHEY_DUPLEX, 0.8, kolor, 2)
        y_historia += 30

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(ramka, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Rozpoznawanie Ruchu na Żywo", ramka)

    if cv2.waitKey(5) & 0xFF == ord('q'): break

kamera.release()
hands.close()
cv2.destroyAllWindows()