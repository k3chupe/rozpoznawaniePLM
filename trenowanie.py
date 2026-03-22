import os
import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. KONFIGURACJA
# ==========================================
FOLDER_Z_DANYMI = "lepsze_dane" # Zmień na swój folder z obrazkami

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================
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
    kat = math.atan2(najdalszy_punkt[1], najdalszy_punkt[0]) / math.pi
    
    # Krok 6: Spłaszczamy 21x2 do 42 liczb i dodajemy kąt na koniec
    cechy = punkty_znormalizowane.flatten().tolist()
    cechy.append(kat)
    
    return np.array(cechy)

# ==========================================
# 3. ZBIERANIE DANYCH
# ==========================================
dane = []
etykiety = []

print(f"Wczytywanie i analizowanie obrazków z folderu '{FOLDER_Z_DANYMI}'...")
for plik in os.listdir(FOLDER_Z_DANYMI):
    litera = plik[0].upper()
    sciezka = os.path.join(FOLDER_Z_DANYMI, plik)
    
    obraz = cv2.imread(sciezka)
    if obraz is None: continue
        
    obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
    
    # WERSJA 1: ORYGINALNY OBRAZ
    wynik = hands.process(obraz_rgb)
    if wynik.multi_hand_landmarks:
        cechy = unifikuj_punkty(wynik.multi_hand_landmarks[0])
        dane.append(cechy)
        etykiety.append(litera)
        
    # WERSJA 2: ODBICIE LUSTRZANE (Druga ręka)
    obraz_odbity = cv2.flip(obraz_rgb, 1) # 1 oznacza odbicie w poziomie
    wynik_odbity = hands.process(obraz_odbity)
    
    if wynik_odbity.multi_hand_landmarks:
        cechy_odbite = unifikuj_punkty(wynik_odbity.multi_hand_landmarks[0])
        dane.append(cechy_odbite)
        etykiety.append(litera)

dane = np.array(dane)
hands.close()

# ==========================================
# 4. PRZYGOTOWANIE ETYKIET I WAG KLAS
# ==========================================
le = LabelEncoder()
etykiety_int = le.fit_transform(etykiety)

# Wyciągamy unikalne klasy i ich liczebność
klasy, ilosci = np.unique(etykiety_int, return_counts=True)

# Wyliczamy wagi algorytmem 'balanced'
wagi = compute_class_weight('balanced', classes=klasy, y=etykiety_int)
class_weight_dict = dict(zip(klasy, wagi))

# Wypisanie statystyk
print("\n" + "="*50)
print("   STATYSTYKI KLAS I ZAPROPONOWANE WAGI")
print("="*50)
print(f"{'KLASA':<10} | {'ILOŚĆ ZDJĘĆ':<15} | {'ZAPROPONOWANA WAGA'}")
print("-" * 50)

for cls, count, weight in zip(klasy, ilosci, wagi):
    nazwa_klasy = le.classes_[cls]
    if str(nazwa_klasy) == '0':
        nazwa_klasy = '0 (Błędy)'
    print(f"{nazwa_klasy:<10} | {count:<15} | {weight:.4f}")

print("="*50 + "\n")

# Kodowanie etykiet (One-Hot) dla sieci neuronowej Keras
lb = LabelBinarizer()
etykiety_one_hot = lb.fit_transform(etykiety)

# Podział danych (z parametrem stratify dla zachowania proporcji!)
X_train, X_test, y_train, y_test = train_test_split(
    dane, etykiety_one_hot, test_size=0.2, random_state=42, stratify=etykiety_int
)

print(f"Zebrano {len(dane)} próbek. Liczba cech: {dane.shape[1]}")

# ==========================================
# 5. BUDOWA I KONFIGURACJA SIECI NEURONOWEJ
# ==========================================
model = Sequential([
    Dense(256, activation='relu', input_shape=(43,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(lb.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 6. CALLBACKI (ZAPISYWANIE NAJLEPSZEGO MODELU)
# ==========================================
nazwa_modelu = "model_gesty_punkty.keras"

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=150,               # Czekaj 30 epok bez poprawy przed przerwaniem
    restore_best_weights=True, # Przywróć najlepsze wagi na koniec
    verbose=1
)

checkpoint = ModelCheckpoint(
    nazwa_modelu, 
    monitor='val_loss', 
    save_best_only=True,       # Zapisz tylko, gdy pobito rekord
    mode='min', 
    verbose=1
)

# ==========================================
# 7. TRENOWANIE SIECI
# ==========================================
print("\nRozpoczynam trenowanie sieci (z monitorowaniem postępów)...")
model.fit(
    X_train, y_train, 
    epochs=2000, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint]
)

# Zapisanie etykiet do pliku pkl (model zapisał się sam dzięki ModelCheckpoint)
with open("etykiety_punkty.pkl", "wb") as f:
    pickle.dump(lb, f)

print(f"\nTrening zakończony! Najlepsza wersja modelu została zapisana jako '{nazwa_modelu}'.")