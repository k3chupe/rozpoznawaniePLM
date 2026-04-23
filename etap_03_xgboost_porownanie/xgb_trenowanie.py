import os
import cv2
import math
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import pickle

# Konfiguracja
FOLDER_Z_DANYMI = "../lepsze_dane"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Funkcja unifikująca 21 punktów do 43 liczb wg Twojego opisu (BEZ ZMIAN)
def unifikuj_punkty(landmarks):
    punkty = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    nadgarstek = punkty[0]
    punkty_przesuniete = punkty - nadgarstek
    
    odleglosci = np.linalg.norm(punkty_przesuniete, axis=1)
    indeks_najdalszego = np.argmax(odleglosci)
    max_odleglosc = odleglosci[indeks_najdalszego]
    
    if max_odleglosc == 0:
        max_odleglosc = 1.0 
        
    punkty_znormalizowane = punkty_przesuniete / max_odleglosc
    
    najdalszy_punkt = punkty_przesuniete[indeks_najdalszego]
    kat = math.atan2(najdalszy_punkt[1], najdalszy_punkt[0]) / math.pi
    
    cechy = punkty_znormalizowane.flatten().tolist()
    cechy.append(kat)
    
    return np.array(cechy)

# Zbieranie danych (BEZ ZMIAN)
dane = []
etykiety = []

print("Wczytywanie i analizowanie obrazków (z odbiciami lustrzanymi)...")
for plik in os.listdir(FOLDER_Z_DANYMI):
    litera = plik[0].upper()
    sciezka = os.path.join(FOLDER_Z_DANYMI, plik)
    
    obraz = cv2.imread(sciezka)
    if obraz is None: continue
        
    obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
    
    # ORYGINALNY OBRAZ
    wynik = hands.process(obraz_rgb)
    if wynik.multi_hand_landmarks:
        cechy = unifikuj_punkty(wynik.multi_hand_landmarks[0])
        dane.append(cechy)
        etykiety.append(litera)
        
    # ODBICIE LUSTRZANE
    obraz_odbity = cv2.flip(obraz_rgb, 1)
    wynik_odbity = hands.process(obraz_odbity)
    
    if wynik_odbity.multi_hand_landmarks:
        cechy_odbite = unifikuj_punkty(wynik_odbity.multi_hand_landmarks[0])
        dane.append(cechy_odbite)
        etykiety.append(litera)

dane = np.array(dane)
hands.close()

# ZMIANA 1: Kodowanie etykiet dla XGBoost (A -> 0, B -> 1, C -> 2 itd.)
le = LabelEncoder()
etykiety_encoded = le.fit_transform(etykiety)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(dane, etykiety_encoded, test_size=0.2, random_state=42, stratify=etykiety_encoded)

print(f"Zebrano {len(dane)} próbek. Liczba cech: {dane.shape[1]}")
print(f"Liczba wykrytych klas (gestów): {len(le.classes_)}")

wagi = compute_sample_weight(class_weight='balanced', y=y_train)

# ZMIANA 2: Definicja modelu XGBoost
print("Rozpoczynam trenowanie modelu XGBoost...")
model = xgb.XGBClassifier(
    objective='multi:softprob',  # dla klasyfikacji wieloklasowej
    num_class=len(le.classes_),  # liczba unikalnych gestów
    eval_metric='mlogloss',      # funkcja błędu
    max_depth=6,                 # głębokość drzew (6 to dobry punkt wyjścia)
    learning_rate=0.1,           # szybkość uczenia
    n_estimators=1000,           # liczba drzew (odpowiednik epok)
    random_state=42,
    n_jobs=-1,                   # użyj wszystkich rdzeni procesora dla przyspieszenia
    early_stopping_rounds=50
)

# Trenowanie modelu (podajemy zbiór testowy do ewaluacji w trakcie uczenia)
model.fit(
    X_train, y_train, 
    sample_weight=wagi,
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=50 # Wyświetlaj postęp co 50 drzew
)

# Ocena modelu na zbiorze testowym
print("\nOcenianie modelu...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- WYNIKI XGBOOST ---")
print(f"Dokładność (Accuracy): {accuracy * 100:.2f}%")
# Wyświetlamy raport używając prawdziwych nazw liter z powrotem
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ZMIANA 3: Zapisywanie z nowymi nazwami
# XGBoost najlepiej zapisywać w formacie JSON
model.save_model("model_gesty_xgboost.json")
with open("etykiety_xgboost.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nTrening zakończony pomyślnie! Zapisano pliki: model_gesty_xgboost.json oraz etykiety_xgboost.pkl")