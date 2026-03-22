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

from keras.callbacks import ReduceLROnPlateau

# ==========================================
# 5. FABRYKA MODELI (DLA TUNERA) - ROZSZERZONA
# ==========================================
def buduj_model(hp):
    model = Sequential()
    
    # Warstwa 1 (Poszerzyliśmy zakres do 1024 neuronów!)
    model.add(Dense(
        units=hp.Int('neurony_warstwa_1', min_value=128, max_value=1024, step=128), 
        input_shape=(X_train.shape[1],)
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.6, step=0.1)))
    
    # Warstwa 2 (Poszerzony zakres do 512 neuronów)
    model.add(Dense(
        units=hp.Int('neurony_warstwa_2', min_value=64, max_value=512, step=64)
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.6, step=0.1)))
    
    # OPCJONALNA WARSTWA 3 (Tuner zdecyduje, czy warto ją dodać)
    if hp.Boolean('dodaj_warstwe_3'):
        model.add(Dense(
            units=hp.Int('neurony_warstwa_3', min_value=32, max_value=256, step=32)
        ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Warstwa wyjściowa
    model.add(Dense(len(lb.classes_), activation='softmax'))
    
    # Algorytm sprawdzi wszystkie ułamki z tej skali zamiast tylko 3 konkretnych liczb
    szybkosc_uczenia = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=Adam(learning_rate=szybkosc_uczenia), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# ==========================================
# 6. SZUKANIE NAJLEPSZEJ SIECI (BAYESIAN OPTIMIZATION)
# ==========================================
print("Rozpoczynam całonocne poszukiwania za pomocą Optymalizacji Bayesowskiej...")

tuner = kt.Hyperband(
    buduj_model,
    objective='val_loss',
    max_epochs=200,          # Maksymalny czas tylko dla finalistów (Twój wielki finał)
    factor=4,                # Zostawia górne 20% na każdym etapie eliminacji
    hyperband_iterations=3,  # Powtarza cały ten wielki "turniej" 3 razy (dla pewności)
    directory='moje_poszukiwania_noc',
    project_name='gesty_hyperband_turniej'
)

# W Hyperband nie musimy ustawiać sztucznie małej cierpliwości (patience)!
# Sam algorytm ucina sieci zgodnie z "drabinką turniejową".
# Ten EarlyStopping zabezpiecza po prostu finalistów przed przetrenowaniem.
tuner_early_stop = EarlyStopping(monitor='val_loss', patience=30)

tuner.search(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[tuner_early_stop],
    verbose=1
)

najlepsze_hiperparametry = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n" + "="*50)
print("ZAKOŃCZONO SZUKANIE! OTO NAJLEPSZA ZNALEZIONA ARCHITEKTURA:")
print(f"- Neurony w 1. warstwie: {najlepsze_hiperparametry.get('neurony_warstwa_1')}")
print(f"- Neurony w 2. warstwie: {najlepsze_hiperparametry.get('neurony_warstwa_2')}")
if najlepsze_hiperparametry.get('dodaj_warstwe_3'):
    print(f"- Zdecydowano się DODAĆ 3. warstwę z neuronami: {najlepsze_hiperparametry.get('neurony_warstwa_3')}")
else:
    print("- Zdecydowano, że 3. warstwa NIE JEST POTRZEBNA.")
print(f"- Prędkość uczenia: {najlepsze_hiperparametry.get('learning_rate')}")
print("="*50 + "\n")

# ==========================================
# 7. OSTATECZNY TRENING ZWYCIĘZCY ("SPUSZCZENIE ZE SMYCZY")
# ==========================================
print("Rozpoczynam potężny trening ostateczny najlepszego modelu...")

najlepszy_model = tuner.hypermodel.build(najlepsze_hiperparametry)
nazwa_modelu = "model_gesty_punkty_v2_nocny.keras"

# STRAŻNIK 1: Zatrzymuje i cofa do najlepszego momentu (cierpliwość = 60 epok)
final_early_stop = EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True, verbose=1)

# STRAŻNIK 2: Precyzyjne parkowanie. Jeśli wynik stoi przez 15 epok, zwalnia uczenie o połowę
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1)

checkpoint = ModelCheckpoint(nazwa_modelu, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

najlepszy_model.fit(
    X_train, y_train, 
    epochs=2000,          # Ogromny zapas epok - strażnik EarlyStop zatrzyma to w idealnym momencie!
    batch_size=32, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[final_early_stop, reduce_lr, checkpoint]
)

with open("etykiety_punkty_v2_nocny.pkl", "wb") as f:
    pickle.dump(lb, f)

print(f"\nUkończono! Rewelacyjnie wytrenowana sieć czeka w pliku '{nazwa_modelu}'.")