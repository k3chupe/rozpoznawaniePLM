import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import time


KAMERA_SZEROKOSC = 1920
KAMERA_WYSOKOSC = 1080
# Ładowanie MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("Ładowanie modelu...")
model = tf.keras.models.load_model("model_gesty_punkty.keras")
with open("etykiety_punkty.pkl", "rb") as f:
    lb = pickle.load(f)

# Funkcja unifikująca 21 punktów do 43 liczb
def unifikuj_punkty(landmarks):
    punkty = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    nadgarstek = punkty[0]
    punkty_przesuniete = punkty - nadgarstek
    
    odleglosci = np.linalg.norm(punkty_przesuniete, axis=1)
    indeks_najdalszego = np.argmax(odleglosci)
    max_odleglosc = odleglosci[indeks_najdalszego]
    if max_odleglosc == 0: max_odleglosc = 1.0
        
    punkty_znormalizowane = punkty_przesuniete / max_odleglosc
    
    najdalszy_punkt = punkty_przesuniete[indeks_najdalszego]
    kat = math.atan2(najdalszy_punkt[1], najdalszy_punkt[0]) / math.pi
    
    cechy = punkty_znormalizowane.flatten().tolist()
    cechy.append(kat)
    return np.array(cechy)

# Funkcja pomocnicza do czytelniejszego wyświetlania zera
def formatuj_nazwe(litera):
    if str(litera) == '0':
        return "0 (ZERO)"
    return str(litera)

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
            cechy_dla_modelu = np.reshape(cechy, (1, 43))
            
            # Pobranie przewidywań
            przewidywania = model(cechy_dla_modelu, training=False).numpy()[0]
            
            # np.argsort sortuje rosnąco, więc odwracamy tablicę [::-1], żeby mieć malejąco
            posortowane_indeksy = np.argsort(przewidywania)[::-1]
            
            # --- 1. GŁÓWNY WYNIK (Najbardziej prawdopodobny) ---
            glowny_indeks = posortowane_indeksy[0]
            glowne_prawd = przewidywania[glowny_indeks]
            glowna_litera = formatuj_nazwe(lb.classes_[glowny_indeks])
            
            if glowne_prawd > 0.6: 
                tekst = f"Gest: {glowna_litera} ({glowne_prawd * 100:.1f}%)"
                # Tło pod główny napis, żeby nie zlewał się z obrazem
                cv2.rectangle(ramka, (40, 45), (450, 95), (0,0,0), -1)
                cv2.putText(ramka, tekst, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # --- 2. WYNIKI ALTERNATYWNE (Inne opcje > 5%) ---
            y_offset = 125 # Zmienna kontrolująca, jak nisko rysować kolejne linijki
            
            for i in range(1, len(posortowane_indeksy)):
                indeks_alt = posortowane_indeksy[i]
                prawd_alt = przewidywania[indeks_alt]
                
                # Jeśli prawdopodobieństwo spadnie poniżej 5% (0.05), przerywamy pętlę
                if prawd_alt < 0.05:
                    break
                    
                litera_alt = formatuj_nazwe(lb.classes_[indeks_alt])
                
                # Dobór koloru: >20% Żółty, pomiędzy 5% a 20% Czerwony (BGR)
                if prawd_alt > 0.20:
                    kolor_alt = (0, 255, 255) # Żółty
                else:
                    kolor_alt = (0, 0, 255) # Czerwony
                
                tekst_alt = f"Moze to: {litera_alt} ({prawd_alt * 100:.1f}%)"
                
                # Cień pod mniejszym tekstem
                cv2.putText(ramka, tekst_alt, (52, y_offset+2), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                # Właściwy mniejszy tekst
                cv2.putText(ramka, tekst_alt, (50, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.6, kolor_alt, 1, cv2.LINE_AA)
                
                y_offset += 30 # Przesuwamy się w dół na kolejną ewentualną linijkę

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(ramka, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Rozpoznawanie Gestów - AI na żywo", ramka)

    if cv2.waitKey(5) & 0xFF == ord('q'): break

kamera.release()
hands.close()
cv2.destroyAllWindows()