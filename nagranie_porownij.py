import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
import xgboost as xgb
import pickle
import time

# ---------------------------------------------------------
# 1. KONFIGURACJA PLIKÓW
# ---------------------------------------------------------
PLIK_WEJSCIOWY = "moje_nagranie.mp4"    # Wpisz nazwę swojego nagrania
PLIK_WYJSCIOWY = "wynik_porownanie.mp4" # Jak ma się nazywać gotowy plik

# ---------------------------------------------------------
# 2. ŁADOWANIE MODELI
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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
# 3. FUNKCJE POMOCNICZE
# ---------------------------------------------------------
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

def formatuj_nazwe(litera):
    if str(litera) == '0': return "0 (Brak / Zly gest)"
    return str(litera)

def rysuj_statystyki(ramka, przewidywania, nazwy_klas, nazwa_modelu, start_x, start_y, kolor_tytulu):
    posortowane_indeksy = np.argsort(przewidywania)[::-1]
    
    glowny_indeks = posortowane_indeksy[0]
    glowne_prawd = przewidywania[glowny_indeks]
    glowna_litera = formatuj_nazwe(nazwy_klas[glowny_indeks])
    
    cv2.putText(ramka, nazwa_modelu, (start_x, start_y - 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, kolor_tytulu, 1, cv2.LINE_AA)
    
    if glowne_prawd > 0.6: 
        tekst = f"Gest: {glowna_litera} ({glowne_prawd * 100:.1f}%)"
        cv2.rectangle(ramka, (start_x - 5, start_y - 25), (start_x + 350, start_y + 10), (0,0,0), -1)
        cv2.putText(ramka, tekst, (start_x, start_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    y_offset = start_y + 35 
    for i in range(1, len(posortowane_indeksy)):
        indeks_alt = posortowane_indeksy[i]
        prawd_alt = przewidywania[indeks_alt]
        
        if prawd_alt < 0.05: break
            
        litera_alt = formatuj_nazwe(nazwy_klas[indeks_alt])
        kolor_alt = (0, 255, 255) if prawd_alt > 0.20 else (0, 0, 255)
        
        tekst_alt = f"Moze to: {litera_alt} ({prawd_alt * 100:.1f}%)"
        cv2.putText(ramka, tekst_alt, (start_x + 2, y_offset + 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(ramka, tekst_alt, (start_x, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, kolor_alt, 1, cv2.LINE_AA)
        y_offset += 25 


# ---------------------------------------------------------
# 4. PRZETWARZANIE WIDEO
# ---------------------------------------------------------
print(f"Otwieranie pliku: {PLIK_WEJSCIOWY}...")
wideo = cv2.VideoCapture(PLIK_WEJSCIOWY)

# Sprawdzenie, czy plik istnieje
if not wideo.isOpened():
    print(f"BŁĄD: Nie można otworzyć pliku '{PLIK_WEJSCIOWY}'. Sprawdź nazwę i ścieżkę.")
    exit()

# Pobieranie parametrów z oryginalnego filmu
szerokosc = int(wideo.get(cv2.CAP_PROP_FRAME_WIDTH))
wysokosc = int(wideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(wideo.get(cv2.CAP_PROP_FPS))
calkowita_liczba_klatek = int(wideo.get(cv2.CAP_PROP_FRAME_COUNT))

# Konfiguracja zapisywania (kodek mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(PLIK_WYJSCIOWY, fourcc, fps, (szerokosc, wysokosc))

print(f"Rozpoczynam przetwarzanie. Plik wyjściowy to: {PLIK_WYJSCIOWY}")
print(f"Rozdzielczość: {szerokosc}x{wysokosc}, FPS: {fps}, Klatek: {calkowita_liczba_klatek}")

licznik_klatek = 0
start_time = time.time()

while True:
    ret, ramka = wideo.read()
    if not ret: 
        break # Koniec filmu
        
    # Odkomentuj poniższą linijkę, jeśli nagrałeś się "jak w lustrze" (np. przednią kamerką) i gesty są na odwrót
    # ramka = cv2.flip(ramka, 1)
    
    ramka_rgb = cv2.cvtColor(ramka, cv2.COLOR_BGR2RGB)
    wynik = hands.process(ramka_rgb)
    
    if wynik.multi_hand_landmarks:
        for hand_landmarks in wynik.multi_hand_landmarks:
            mp_drawing.draw_landmarks(ramka, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cechy = unifikuj_punkty(hand_landmarks)
            cechy_dla_modelu = np.reshape(cechy, (1, 43))
            
            przewidywania_keras = model_keras(cechy_dla_modelu, training=False).numpy()[0]
            przewidywania_xgb = model_xgb.predict_proba(cechy_dla_modelu)[0] 
            
            rysuj_statystyki(
                ramka, przewidywania_keras, lb_keras.classes_, 
                "KERAS", start_x=20, start_y=80, kolor_tytulu=(0, 255, 0)
            )
            
            rysuj_statystyki(
                ramka, przewidywania_xgb, le_xgb.classes_, 
                "XGBOOST", start_x=szerokosc - 350, start_y=80, kolor_tytulu=(255, 100, 0)
            )

    # Zapisanie przetworzonej klatki do nowego filmu
    out.write(ramka)
    
    # Wyświetlanie postępu w konsoli co 30 klatek
    licznik_klatek += 1
    if licznik_klatek % 30 == 0:
        procent = (licznik_klatek / calkowita_liczba_klatek) * 100
        print(f"Przetworzono {licznik_klatek}/{calkowita_liczba_klatek} klatek ({procent:.1f}%)")

# Zamykanie wszystkiego
wideo.release()
out.release()
hands.close()

czas_trwania = time.time() - start_time
print(f"\nSukces! Film zapisany jako '{PLIK_WYJSCIOWY}'. Przetwarzanie zajęło {czas_trwania:.1f} sekund.")