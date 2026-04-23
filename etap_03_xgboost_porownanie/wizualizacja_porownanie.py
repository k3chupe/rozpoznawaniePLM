import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
import xgboost as xgb
import pickle
import time

# ---------------------------------------------------------
# 1. KONFIGURACJA I ŁADOWANIE MODELI
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("Ładowanie modelu Keras (Sieć Neuronowa)...")
model_keras = tf.keras.models.load_model("model_gesty_punkty.keras")
with open("etykiety_punkty.pkl", "rb") as f:
    lb_keras = pickle.load(f)

print("Ładowanie modelu XGBoost...")
model_xgb = xgb.XGBClassifier()
model_xgb.load_model("model_gesty_xgboost.json")
with open("etykiety_xgboost.pkl", "rb") as f:
    le_xgb = pickle.load(f)

# ---------------------------------------------------------
# 2. FUNKCJE POMOCNICZE
# ---------------------------------------------------------
def cechy_statyczne_2d(landmarks):
    # Wejście: 21 punktów 2D (x, y) z MediaPipe Solutions API
    # Wyjście: wektor 43 float (42 współrzędne znormalizowane + kąt atan2) - używane przez modele etap_02/03
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

def formatuj_nazwe(litera):
    if str(litera) == '0': return "0 (ZERO)"
    return str(litera)

# Funkcja rysująca panele informacyjne dla każdego z modeli
def rysuj_statystyki(ramka, przewidywania, nazwy_klas, nazwa_modelu, start_x, start_y, kolor_tytulu):
    posortowane_indeksy = np.argsort(przewidywania)[::-1]
    
    glowny_indeks = posortowane_indeksy[0]
    glowne_prawd = przewidywania[glowny_indeks]
    glowna_litera = formatuj_nazwe(nazwy_klas[glowny_indeks])
    
    # Rysowanie nazwy modelu (Keras lub XGBoost)
    cv2.putText(ramka, nazwa_modelu, (start_x, start_y - 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, kolor_tytulu, 1, cv2.LINE_AA)
    
    # Rysowanie głównego wyniku (jeśli pewność > 60%)
    if glowne_prawd > 0.6: 
        tekst = f"Gest: {glowna_litera} ({glowne_prawd * 100:.1f}%)"
        # Czarne, lekko przezroczyste tło pod tekst
        cv2.rectangle(ramka, (start_x - 5, start_y - 25), (start_x + 300, start_y + 10), (0,0,0), -1)
        cv2.putText(ramka, tekst, (start_x, start_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Rysowanie alternatyw (wyniki > 5%)
    y_offset = start_y + 35 
    for i in range(1, len(posortowane_indeksy)):
        indeks_alt = posortowane_indeksy[i]
        prawd_alt = przewidywania[indeks_alt]
        
        if prawd_alt < 0.05:
            break
            
        litera_alt = formatuj_nazwe(nazwy_klas[indeks_alt])
        kolor_alt = (0, 255, 255) if prawd_alt > 0.20 else (0, 0, 255)
        
        tekst_alt = f"Moze to: {litera_alt} ({prawd_alt * 100:.1f}%)"
        cv2.putText(ramka, tekst_alt, (start_x + 2, y_offset + 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(ramka, tekst_alt, (start_x, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, kolor_alt, 1, cv2.LINE_AA)
        
        y_offset += 25 


# ---------------------------------------------------------
# 3. GŁÓWNA PĘTLA PROGRAMU
# ---------------------------------------------------------
kamera = cv2.VideoCapture(0)
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
            
            cechy = cechy_statyczne_2d(hand_landmarks)
            cechy_dla_modelu = np.reshape(cechy, (1, 43))
            
            # --- PREDYKCJA KERAS ---
            przewidywania_keras = model_keras(cechy_dla_modelu, training=False).numpy()[0]
            
            # --- PREDYKCJA XGBOOST ---
            # predict_proba zwraca tablicę 2D, bierzemy pierwszy [0] element
            przewidywania_xgb = model_xgb.predict_proba(cechy_dla_modelu)[0] 
            
            # --- RYSOWANIE WYNIKÓW ---
            # Keras po lewej stronie (start_x = 20, zielony tytuł)
            rysuj_statystyki(
                ramka, przewidywania_keras, lb_keras.classes_, 
                "KERAS (Siec Neuronowa)", start_x=20, start_y=80, kolor_tytulu=(0, 255, 0)
            )
            
            # XGBoost po prawej stronie (start_x = 340, niebieski tytuł)
            rysuj_statystyki(
                ramka, przewidywania_xgb, le_xgb.classes_, 
                "XGBOOST (Drzewa)", start_x=340, start_y=80, kolor_tytulu=(255, 100, 0)
            )

    # Licznik FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(ramka, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Keras vs XGBoost - Starcie AI", ramka)

    if cv2.waitKey(5) & 0xFF == ord('q'): break

kamera.release()
hands.close()
cv2.destroyAllWindows()