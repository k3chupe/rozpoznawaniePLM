import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import keras_tuner as kt
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. KONFIGURACJA I POBRANIE MODELU MEDIAPIPE
# ==========================================
FOLDER_Z_DANYMI = "lepsze_dane" 
MODEL_TASK_PATH = "hand_landmarker.task"

# Automatyczne pobieranie modelu dla nowego API MediaPipe
if not os.path.exists(MODEL_TASK_PATH):
    print(f"Pobieranie wymaganego modelu MediaPipe ({MODEL_TASK_PATH})...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_TASK_PATH)
    print("Pobieranie zakończone!")

# Konfiguracja nowego MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================
def unifikuj_punkty(landmarks, handedness_category):
    """
    Ekstrakcja punktów 3D, normalizacja i dodanie informacji o ręce.
    Dostosowane do nowego formatu danych MediaPipe Tasks API.
    """
    # Nowe API przechowuje współrzędne bezpośrednio jako atrybuty obiektu
    punkty = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    nadgarstek = punkty[0]
    
    punkty_przesuniete = punkty - nadgarstek
    
    odleglosci = np.linalg.norm(punkty_przesuniete, axis=1)
    max_odleglosc = np.max(odleglosci)
    if max_odleglosc == 0: max_odleglosc = 1.0
        
    punkty_znormalizowane = punkty_przesuniete / max_odleglosc
    cechy = punkty_znormalizowane.flatten().tolist()
    
    # Nowe API zwraca 'Left' lub 'Right' w category_name
    etykieta_reki = handedness_category.category_name
    cechy.append(1.0 if etykieta_reki == 'Right' else 0.0)
    
    return np.array(cechy)

def obroc_obraz(obraz, kat):
    (h, w) = obraz.shape[:2]
    srodek = (w // 2, h // 2)
    macierz = cv2.getRotationMatrix2D(srodek, kat, 1.0)
    return cv2.warpAffine(obraz, macierz, (w, h))

def analizuj_i_dodaj(obraz_rgb, litera, dane_lista, etykiety_lista):
    # Nowe API wymaga konwersji macierzy NumPy na obiekt mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=obraz_rgb)
    wynik = detector.detect(mp_image)
    
    # Sprawdzamy czy wykryto dłoń (hand_landmarks) i jej typ (handedness)
    if wynik.hand_landmarks and wynik.handedness:
        # Pobieramy pierwszą wykrytą dłoń ([0]) i jej pierwszą klasyfikację ([0][0])
        cechy = unifikuj_punkty(wynik.hand_landmarks[0], wynik.handedness[0][0])
        dane_lista.append(cechy)
        etykiety_lista.append(litera)

# ==========================================
# 3. ZBIERANIE DANYCH I AUGMENTACJA
# ==========================================
dane = []
etykiety = []

print(f"Wczytywanie i analizowanie obrazków z podfolderów w '{FOLDER_Z_DANYMI}'...")

if not os.path.exists(FOLDER_Z_DANYMI):
    raise FileNotFoundError(f"Nie znaleziono folderu {FOLDER_Z_DANYMI}! Utwórz go i dodaj podfoldery z klasami.")

for folder_klasy in os.listdir(FOLDER_Z_DANYMI):
    sciezka_klasy = os.path.join(FOLDER_Z_DANYMI, folder_klasy)
    if not os.path.isdir(sciezka_klasy): continue
        
    litera = folder_klasy.upper()
    
    for plik in os.listdir(sciezka_klasy):
        sciezka_obrazu = os.path.join(sciezka_klasy, plik)
        obraz = cv2.imread(sciezka_obrazu)
        
        if obraz is None: continue
        obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
        
        analizuj_i_dodaj(obraz_rgb, litera, dane, etykiety)
        
        obraz_odbity = cv2.flip(obraz_rgb, 1)
        analizuj_i_dodaj(obraz_odbity, litera, dane, etykiety)
        
        obraz_rot_lewo = obroc_obraz(obraz_rgb, -15)
        analizuj_i_dodaj(obraz_rot_lewo, litera, dane, etykiety)
        
        obraz_rot_prawo = obroc_obraz(obraz_rgb, 15)
        analizuj_i_dodaj(obraz_rot_prawo, litera, dane, etykiety)

dane = np.array(dane)

if len(dane) == 0:
    raise ValueError("Nie wyekstrahowano żadnych cech! Sprawdź strukturę folderów i jakość zdjęć.")

print(f"Wygenerowano {len(dane)} próbek (w tym augmentacja) z wektorami o długości {dane.shape[1]}.")

# ==========================================
# 4. PRZYGOTOWANIE ETYKIET I WAG KLAS
# ==========================================
le = LabelEncoder()
etykiety_int = le.fit_transform(etykiety)

klasy, ilosci = np.unique(etykiety_int, return_counts=True)
wagi = compute_class_weight('balanced', classes=klasy, y=etykiety_int)
class_weight_dict = dict(zip(klasy, wagi))

print("\n" + "="*50)
print("   STATYSTYKI KLAS I ZAPROPONOWANE WAGI")
print("="*50)
for cls, count, weight in zip(klasy, ilosci, wagi):
    print(f"{le.classes_[cls]:<10} | {count:<15} | {weight:.4f}")
print("="*50 + "\n")

lb = LabelBinarizer()
etykiety_one_hot = lb.fit_transform(etykiety)

X_train, X_test, y_train, y_test = train_test_split(
    dane, etykiety_one_hot, test_size=0.2, random_state=42, stratify=etykiety_int
)

# ==========================================
# 5. FABRYKA MODELI (DLA TUNERA)
# ==========================================
def buduj_model(hp):
    model = Sequential()
    
    model.add(Dense(
        units=hp.Int('neurony_warstwa_1', min_value=64, max_value=512, step=64), 
        input_shape=(X_train.shape[1],)
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(Dense(
        units=hp.Int('neurony_warstwa_2', min_value=32, max_value=256, step=32)
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(Dense(len(lb.classes_), activation='softmax'))
    
    szybkosc_uczenia = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=Adam(learning_rate=szybkosc_uczenia), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# ==========================================
# 6. SZUKANIE NAJLEPSZEJ SIECI (HYPERBAND)
# ==========================================
print("Rozpoczynam poszukiwania architektur za pomocą algorytmu Hyperband...")

tuner = kt.Hyperband(
    buduj_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='moje_poszukiwania',
    project_name='rozpoznawanie_gestow_v2'
)

tuner_early_stop = EarlyStopping(monitor='val_loss', patience=10)

tuner.search(
    X_train, y_train, 
    epochs=50, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[tuner_early_stop],
    verbose=1
)

najlepsze_hiperparametry = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n" + "="*50)
print("ZNALEZIONO NAJLEPSZĄ ARCHITEKTURĘ SIECI!")
print("="*50 + "\n")

# ==========================================
# 7. OSTATECZNY TRENING ZWYCIĘZCY
# ==========================================
print("Rozpoczynam ostateczny trening na najlepszym modelu...")

najlepszy_model = tuner.hypermodel.build(najlepsze_hiperparametry)
nazwa_modelu = "model_gesty_punkty_v2.keras"

final_early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(nazwa_modelu, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

najlepszy_model.fit(
    X_train, y_train, 
    epochs=1000, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[final_early_stop, checkpoint]
)

with open("etykiety_punkty_v2.pkl", "wb") as f:
    pickle.dump(lb, f)

print(f"\nGotowe! Najlepsza sieć zapisana jako '{nazwa_modelu}'.")