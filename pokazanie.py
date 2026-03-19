import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

# Ładowanie MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("Ładowanie modelu...")
model = tf.keras.models.load_model("model_gesty_punkty.keras")
with open("etykiety_punkty.pkl", "rb") as f:
    lb = pickle.load(f)

# Ta sama funkcja co w skrypcie treningowym
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

kamera = cv2.VideoCapture(0)
print("Kamera uruchomiona. Wciśnij 'q', aby wyjść.")

while True:
    ret, ramka = kamera.read()
    if not ret: break
        
    # Odbicie lustrzane dla wygody i konwersja do RGB dla MediaPipe
    ramka = cv2.flip(ramka, 1)
    ramka_rgb = cv2.cvtColor(ramka, cv2.COLOR_BGR2RGB)
    
    wynik = hands.process(ramka_rgb)
    
    if wynik.multi_hand_landmarks:
        for hand_landmarks in wynik.multi_hand_landmarks:
            # Rysowanie szkieletu na dłoni dla podglądu
            mp_drawing.draw_landmarks(ramka, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Przekształcenie punktów na 43 liczby
            cechy = unifikuj_punkty(hand_landmarks)
            cechy_dla_modelu = np.reshape(cechy, (1, 43)) # Model oczekuje wymiaru (batch_size, features)
            
            # Predykcja
            przewidywania = model.predict(cechy_dla_modelu, verbose=0)
            indeks = np.argmax(przewidywania)
            prawdopodobienstwo = przewidywania[0][indeks]
            rozpoznana_litera = lb.classes_[indeks]
            
            # Wyświetlanie wyniku
            if prawdopodobienstwo > 0.6: # próg pewności 60%
                tekst = f"Gest: {rozpoznana_litera} ({prawdopodobienstwo * 100:.1f}%)"
                cv2.putText(ramka, tekst, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Rozpoznawanie Gestów - AI na żywo", ramka)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
hands.close()
cv2.destroyAllWindows()