import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

# ==========================================
# KONFIGURACJA MODELU I KAMERY
# ==========================================
SCIEZKA_MODELU = "model_gesty_punkty_v2_nocny.keras"
SCIEZKA_ETYKIET = "etykiety_punkty_v2_nocny.pkl"
PEWNOSC_THRESHOLD = 0.65 
# Rozdzielczość kamery
KAMERA_SZEROKOSC = 1920
KAMERA_WYSOKOSC = 1080
# ==========================================

def obrot_3d(x, y, z, kat_x, kat_y):
    """Funkcja pomocnicza do obrotu punktu w 3D (najpierw oś X, potem oś Y)"""
    # Obrót wokół osi X (pochylenie do przodu/tyłu)
    y1 = y * math.cos(kat_x) - z * math.sin(kat_x)
    z1 = y * math.sin(kat_x) + z * math.cos(kat_x)
    
    # Obrót wokół osi Y (kręcenie w lewo/prawo)
    x2 = x * math.cos(kat_y) + z1 * math.sin(kat_y)
    z2 = -x * math.sin(kat_y) + z1 * math.cos(kat_y)
    
    return x2, y1, z2

def rysuj_wykres_3d(ramka, hand_landmarks, kat_obrotu_y, mp_hands, srodek_x, srodek_y, skala=500):
    """Rysuje dłoń w 3D wraz z obracającą się klatką (siatką współrzędnych)."""
    
    # 1. Rysowanie ciemnoszarego tła pod wykres (dla lepszej widoczności)
    w_tla = 400
    h_tla = 400
    cv2.rectangle(ramka, (srodek_x - w_tla//2, srodek_y - h_tla//2), 
                         (srodek_x + w_tla//2, srodek_y + h_tla//2), 
                         (60, 60, 60), -1)

    # Stały kąt pochylenia siatki (żeby było widać ją "z góry" jak na Twoim screenie)
    kat_pochylenia_x = -0.3 

    # 2. Definicja siatki 3D (sześcianu)
    r = 0.15 # Rozmiar sześcianu (w skali znormalizowanej MediaPipe)
    wierzcholki_siatki = [
        [-r, -r, -r], [ r, -r, -r], [ r,  r, -r], [-r,  r, -r], # Tylna ściana
        [-r, -r,  r], [ r, -r,  r], [ r,  r,  r], [-r,  r,  r]  # Przednia ściana
    ]
    
    krawedzie_siatki = [
        (0,1), (1,2), (2,3), (3,0), # tył
        (4,5), (5,6), (6,7), (7,4), # przód
        (0,4), (1,5), (2,6), (3,7)  # łączenia boczne
    ]

    # Rysowanie siatki
    wierzcholki_2d = []
    for wx, wy, wz in wierzcholki_siatki:
        # Obrót sześcianu
        x_obr, y_obr, z_obr = obrot_3d(wx, wy, wz, kat_pochylenia_x, kat_obrotu_y)
        # Rzutowanie na ekran
        ekran_x = int(x_obr * skala + srodek_x)
        ekran_y = int(y_obr * skala + srodek_y)
        wierzcholki_2d.append((ekran_x, ekran_y))

    for p1_idx, p2_idx in krawedzie_siatki:
        cv2.line(ramka, wierzcholki_2d[p1_idx], wierzcholki_2d[p2_idx], (150, 150, 150), 1, cv2.LINE_AA)

    # 3. Przygotowanie punktów dłoni
    punkty_3d = []
    for lm in hand_landmarks.landmark:
        punkty_3d.append([lm.x, lm.y, lm.z])
    punkty_3d = np.array(punkty_3d)

    # Wyśrodkowanie dłoni, żeby obracała się wokół własnej osi, a nie krawędzi ekranu
    srodek_ciezkosci = np.mean(punkty_3d, axis=0)
    punkty_3d -= srodek_ciezkosci

    punkty_2d_wyswietlanie = []
    for x, y, z in punkty_3d:
        # Obrót punktów dłoni (taki sam jak siatki)
        x_obr, y_obr, z_obr = obrot_3d(x, y, z, kat_pochylenia_x, kat_obrotu_y)
        
        ekran_x = int(x_obr * skala + srodek_x)
        ekran_y = int(y_obr * skala + srodek_y)
        punkty_2d_wyswietlanie.append((ekran_x, ekran_y))

    # 4. Rysowanie kości dłoni (jak na Twoim obrazku)
    for polaczenie in mp_hands.HAND_CONNECTIONS:
        p1 = punkty_2d_wyswietlanie[polaczenie[0]]
        p2 = punkty_2d_wyswietlanie[polaczenie[1]]
        cv2.line(ramka, p1, p2, (200, 200, 200), 2, cv2.LINE_AA) # Szare linie 

    # 5. Rysowanie stawów (zielone punkty jak na obrazku)
    for p in punkty_2d_wyswietlanie:
        cv2.circle(ramka, p, 5, (50, 255, 50), -1, cv2.LINE_AA) # Jasnozielony kolor BGR

def main():
    print("Inicjalizacja środowiska...")
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    print(f"Ładowanie modelu: {SCIEZKA_MODELU}...")
    try:
        model = tf.keras.models.load_model(SCIEZKA_MODELU)
        with open(SCIEZKA_ETYKIET, "rb") as f:
            lb = pickle.load(f)
        
        oczekiwane_cechy = model.input_shape[1]
    except Exception as e:
        print(f"BŁĄD! {e}")
        return

    def unifikuj_punkty(landmarks, tryb_3d=False):
        if tryb_3d:
            punkty = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        else:
            punkty = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
            
        nadgarstek = punkty[0]
        punkty_przesuniete = punkty - nadgarstek
        
        odleglosci = np.linalg.norm(punkty_przesuniete[:, :2], axis=1)
        max_odleglosc = odleglosci[np.argmax(odleglosci)]
        if max_odleglosc == 0: max_odleglosc = 1.0
            
        punkty_znormalizowane = punkty_przesuniete / max_odleglosc
        najdalszy_punkt = punkty_przesuniete[np.argmax(odleglosci)]
        kat = math.atan2(najdalszy_punkt[1], najdalszy_punkt[0]) / math.pi
        
        cechy = punkty_znormalizowane.flatten().tolist()
        cechy.append(kat)
        return np.array(cechy)

    kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Wymuszenie Full HD
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, KAMERA_SZEROKOSC)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, KAMERA_WYSOKOSC)
    
    nazwa_okna = "Rozpoznawanie Gestow - Wersja PRO"
    cv2.namedWindow(nazwa_okna, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(nazwa_okna, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    kat_obrotu_3d = 0.0 
    
    while True:
        ret, ramka = kamera.read()
        if not ret: break
        
        ramka = cv2.flip(ramka, 1)
        wysokosc_ramki, szerokosc_ramki, _ = ramka.shape
        
        ramka_rgb = cv2.cvtColor(ramka, cv2.COLOR_BGR2RGB)
        wynik = hands.process(ramka_rgb)
        
        tekst = "Szukam gestu..."
        kolor = (200, 200, 200)

        if wynik.multi_hand_landmarks:
            for hand_landmarks in wynik.multi_hand_landmarks:
                # Szkielet na rzeczywistej dłoni
                mp_drawing.draw_landmarks(
                    ramka, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Nasz nowy, kręcący się wykres Full HD w prawym górnym rogu
                kat_obrotu_3d += 0.03 # Prędkość obrotu
                rysuj_wykres_3d(
                    ramka, hand_landmarks, kat_obrotu_3d, mp_hands, 
                    srodek_x=szerokosc_ramki - 250, # Odsunięcie od prawej krawędzi (zwiększone dla Full HD)
                    srodek_y=250,                   # Odsunięcie od góry
                    skala=600                       # Większa skala, bo ekran jest Full HD
                )
                
                # Klasyfikacja
                cechy = unifikuj_punkty(hand_landmarks, tryb_3d=(oczekiwane_cechy == 64))
                cechy_dla_modelu = np.reshape(cechy, (1, -1))
                
                przewidywania = model.predict(cechy_dla_modelu, verbose=0)
                indeks = np.argmax(przewidywania)
                prawdopodobienstwo = przewidywania[0][indeks]
                rozpoznana_litera = str(lb.classes_[indeks])

                if prawdopodobienstwo < PEWNOSC_THRESHOLD:
                    tekst = "Szukam gestu..."
                    kolor = (150, 150, 150)
                elif rozpoznana_litera == '0':
                    tekst = "Rozpoznano niepoprawny gest"
                    kolor = (0, 0, 255)
                else:
                    tekst = f"Gest: {rozpoznana_litera} ({prawdopodobienstwo * 100:.0f}%)"
                    kolor = (0, 255, 0) if prawdopodobienstwo > 0.85 else (0, 255, 255)

        # Powiększone elementy interfejsu (żeby były czytelne w Full HD)
        cv2.rectangle(ramka, (20, 30), (700, 110), (0, 0, 0), -1)
        cv2.putText(ramka, tekst, (40, 85), cv2.FONT_HERSHEY_DUPLEX, 1.5, kolor, 3, cv2.LINE_AA)
        
        cv2.imshow(nazwa_okna, ramka)

        klawisz = cv2.waitKey(1) & 0xFF
        if klawisz == ord('q') or klawisz == 27:
            break

    kamera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()