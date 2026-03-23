import os
import cv2
import math
import numpy as np
import mediapipe as mp
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout # Zmieniamy na LSTM!
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# I to zadziała bez błędu:
nazwa_modelu = os.path.join(BASE_DIR, "models", "model_gesty_ruchome.keras")

FOLDER_Z_DANYMI = "nagrania_gestow" # Zmień na swój folder z filmami wideo
MAX_KLATEK_W_GEŚCIE = 30 # Każdy gest kompresujemy/rozciągamy do równych 30 klatek

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# ==========================================
# EKSTRAKCJA 64 CECH (z dodaną osią Z)
# ==========================================
def unifikuj_punkty(landmarks):
    # Krok 1: Wyciągamy X, Y oraz dodajemy Z!
    punkty = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    nadgarstek = punkty[0]
    punkty_przesuniete = punkty - nadgarstek
    
    odleglosci = np.linalg.norm(punkty_przesuniete, axis=1)
    indeks_najdalszego = np.argmax(odleglosci)
    max_odleglosc = odleglosci[indeks_najdalszego]
    if max_odleglosc == 0: max_odleglosc = 1.0
        
    punkty_znormalizowane = punkty_przesuniete / max_odleglosc
    
    najdalszy_punkt = punkty_przesuniete[indeks_najdalszego]
    kat = math.atan2(najdalszy_punkt[1], najdalszy_punkt[0]) / math.pi
    
    cechy = punkty_znormalizowane.flatten().tolist() # Z 21x3 robi się wektor 63 liczb
    cechy.append(kat) # Dodajemy kąt -> Równe 64 cechy!
    return np.array(cechy)

# ==========================================
# NORMALIZACJA CZASOWA WIDEO
# ==========================================
def wyciagnij_sekwencje(sciezka_wideo):
    cap = cv2.VideoCapture(sciezka_wideo)
    sekwencja = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        wyniki = hands.process(frame_rgb)
        
        # Pobieramy punkty, TYLKO gdy dłoń jest widoczna
        if wyniki.multi_hand_landmarks:
            cechy = unifikuj_punkty(wyniki.multi_hand_landmarks[0])
            sekwencja.append(cechy)
            
    cap.release()
    
    # Jeśli w całym filmie dłoń mignęła na mniej niż 5 klatek, to śmieć - ignorujemy
    if len(sekwencja) < 5:
        return None
        
    sekwencja = np.array(sekwencja)
    
    # TZW. TIME NORMALIZATION: Interpolujemy ruch do idealnych 30 klatek
    indeksy_docelowe = np.linspace(0, len(sekwencja) - 1, MAX_KLATEK_W_GEŚCIE).astype(int)
    znormalizowana_sekwencja = sekwencja[indeksy_docelowe]
    
    return znormalizowana_sekwencja

# ==========================================
# ZBIERANIE DANYCH
# ==========================================
dane = []
etykiety = []

print("Ekstrakcja sekwencji z wideo (To może potrwać)...")
for folder_klasy in os.listdir(FOLDER_Z_DANYMI):
    sciezka_klasy = os.path.join(FOLDER_Z_DANYMI, folder_klasy)
    if not os.path.isdir(sciezka_klasy): continue
        
    litera = folder_klasy.upper()
    for plik in os.listdir(sciezka_klasy):
        if plik.lower().endswith(('.mp4', '.avi')):
            sciezka_wideo = os.path.join(sciezka_klasy, plik)
            
            sekwencja_cech = wyciagnij_sekwencje(sciezka_wideo)
            if sekwencja_cech is not None:
                dane.append(sekwencja_cech)
                etykiety.append(litera)

dane = np.array(dane) # Oczekiwany kształt: (IlośćFilmów, 30, 64)
hands.close()

if len(dane) == 0:
    raise ValueError("Nie udało się odczytać żadnych dłoni z filmów!")

print(f"Załadowano {len(dane)} filmów wideo. Kształt danych: {dane.shape}")

# ==========================================
# 4. OBLICZANIE WAG KLAS (BALANSOWANIE NAGRAŃ)
# ==========================================
# Zamieniamy litery (np. 'H', 'Z') na liczby (np. 0, 1), bo tak woli matematyka
le = LabelEncoder()
etykiety_int = le.fit_transform(etykiety)

# Wyciągamy unikalne numery klas i zliczamy, ile razy każda występuje (ile masz nagrań)
klasy, ilosci = np.unique(etykiety_int, return_counts=True)

# Magiczna funkcja! Oblicza wagi tak, by "niedoreprezentowane" nagrania były ważniejsze
wagi = compute_class_weight('balanced', classes=klasy, y=etykiety_int)

# Tworzymy słownik dla Kerasa (np. {0: 1.25, 1: 0.8})
class_weight_dict = dict(zip(klasy, wagi))

# --- WYŚWIETLANIE STATYSTYK DLA INŻYNIERA ---
print("\n" + "="*50)
print("   STATYSTYKI NAGRAŃ WIDEO I WAGI SIECI")
print("="*50)
print(f"{'GEST':<10} | {'ILOŚĆ NAGRAŃ':<15} | {'WAGA DLA SIECI'}")
print("-" * 50)

for cls, count, weight in zip(klasy, ilosci, wagi):
    nazwa_klasy = le.classes_[cls]
    print(f"{nazwa_klasy:<10} | {count:<15} | {weight:.4f}")

print("="*50 + "\n")

# Zamiana na format One-Hot dla sieci neuronowej (np. [1,0,0], [0,1,0])
lb = LabelBinarizer()
etykiety_one_hot = lb.fit_transform(etykiety)

X_train, X_test, y_train, y_test = train_test_split(dane, etykiety_one_hot, test_size=0.2, random_state=42, stratify=etykiety_int)

# ==========================================
# NOWA ARCHITEKTURA MODELU (LSTM do Ruchu)
# ==========================================
model = Sequential([
    # Zauważ input_shape: (30 klatek, 64 cechy)
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(MAX_KLATEK_W_GEŚCIE, 64)),
    Dropout(0.2),
    LSTM(32, return_sequences=False, activation='tanh'), # Druga warstwa zbiera wnioski i je zamyka
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(lb.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# KONFIGURACJA "STRAŻNIKÓW" TRENINGU
# ==========================================
nazwa_modelu = os.path.join(BASE_DIR, "models", "model_gesty_ruchome.keras")

# 1. Główny strażnik - Wczesne Zatrzymanie
early_stop = EarlyStopping(
    monitor='val_loss',        # Obserwuje błąd na zbiorze testowym
    patience=30,               # <--- ZMIANA: Przerywa trening, jeśli przez 30 epok nie ma bicia rekordu!
    restore_best_weights=True, # <--- BARDZO WAŻNE: Po przerwaniu automatycznie cofa model do najlepszej epoki
    verbose=1
)

# 2. Zapisywacz - na bieżąco nadpisuje plik, jeśli padł nowy rekord
checkpoint = ModelCheckpoint(
    nazwa_modelu, 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min', 
    verbose=1
)

# ==========================================
# ODPALENIE TRENINGU
# ==========================================
print("\nRozpoczynam trenowanie modelu ruchu...")

model.fit(
    X_train, y_train, 
    epochs=1000, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,   # <--- TO JEST TEN KLUCZOWY PARAMETR!
    callbacks=[early_stop, checkpoint]
)

with open("etykiety_ruch.pkl", "wb") as f:
    pickle.dump(lb, f)

print(f"\nUkończono! Gotowy do rozpoznawania ruchu z użyciem '{nazwa_modelu}'.")