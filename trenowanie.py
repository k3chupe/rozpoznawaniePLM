import os
import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle

# Konfiguracja
FOLDER_Z_DANYMI = "lepsze_dane" # Zmień na swój folder z obrazkami

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Funkcja unifikująca 21 punktów do 43 liczb wg Twojego opisu
def unifikuj_punkty(landmarks):
    # Krok 1: Wyciągamy x i y do tablicy numpy
    punkty = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    
    # Krok 2: Nadgarstek to punkt 0,0
    nadgarstek = punkty[0]
    punkty_przesuniete = punkty - nadgarstek
    
    # Krok 3: Znajdowanie najdalszego punktu i jego odległości
    odleglosci = np.linalg.norm(punkty_przesuniete, axis=1)
    indeks_najdalszego = np.argmax(odleglosci)
    max_odleglosc = odleglosci[indeks_najdalszego]
    
    if max_odleglosc == 0:
        max_odleglosc = 1.0 # Zabezpieczenie przed dzieleniem przez zero
        
    # Krok 4: Skalowanie punktów, aby najdalszy był w odległości 1 (wartości od -1 do 1)
    punkty_znormalizowane = punkty_przesuniete / max_odleglosc
    
    # Krok 5: Obliczanie kąta wektora do najdalszego punktu
    najdalszy_punkt = punkty_przesuniete[indeks_najdalszego]
    # math.atan2 zwraca kąt od -pi do pi. Dzieląc przez pi, mamy zakres od -1 do 1
    kat = math.atan2(najdalszy_punkt[1], najdalszy_punkt[0]) / math.pi
    
    # Krok 6: Spłaszczamy 21x2 do 42 liczb i dodajemy kąt na koniec
    cechy = punkty_znormalizowane.flatten().tolist()
    cechy.append(kat)
    
    return np.array(cechy)

# Zbieranie danych
dane = []
etykiety = []

print("Wczytywanie i analizowanie obrazków (z odbiciami lustrzanymi)...")
for plik in os.listdir(FOLDER_Z_DANYMI):
    litera = plik[0].upper()
    sciezka = os.path.join(FOLDER_Z_DANYMI, plik)
    
    obraz = cv2.imread(sciezka)
    if obraz is None: continue
        
    obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
    
    # -----------------------------------------
    # WERSJA 1: ORYGINALNY OBRAZ
    # -----------------------------------------
    wynik = hands.process(obraz_rgb)
    if wynik.multi_hand_landmarks:
        cechy = unifikuj_punkty(wynik.multi_hand_landmarks[0])
        dane.append(cechy)
        etykiety.append(litera)
        
    # -----------------------------------------
    # WERSJA 2: ODBICIE LUSTRZANE (Druga ręka)
    # -----------------------------------------
    obraz_odbity = cv2.flip(obraz_rgb, 1) # 1 oznacza odbicie w poziomie
    wynik_odbity = hands.process(obraz_odbity)
    
    if wynik_odbity.multi_hand_landmarks:
        cechy_odbite = unifikuj_punkty(wynik_odbity.multi_hand_landmarks[0])
        dane.append(cechy_odbite)
        etykiety.append(litera)

dane = np.array(dane)
hands.close()

# Kodowanie etykiet (A -> [1,0,0], B -> [0,1,0])
lb = LabelBinarizer()
etykiety = lb.fit_transform(etykiety)

X_train, X_test, y_train, y_test = train_test_split(dane, etykiety, test_size=0.2, random_state=42)

print(f"Zebrano {len(dane)} próbek. Liczba cech: {dane.shape[1]}")

# Architektura Sieci Neuronowej dla danych numerycznych (MLP)
model = Sequential([
    Dense(256, activation='relu', input_shape=(43,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(lb.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Rozpoczynam trenowanie sieci...")
model.fit(X_train, y_train, epochs=700, batch_size=32, validation_data=(X_test, y_test))

# Zapisywanie
model.save("model_gesty_punkty.keras")
with open("etykiety_punkty.pkl", "wb") as f:
    pickle.dump(lb, f)

print("Trening zakończony pomyślnie!")