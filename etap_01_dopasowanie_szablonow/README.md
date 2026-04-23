# Etap 1 — Dopasowanie szablonow

**Cel:** Sprawdzenie czy mozna rozpoznawac gesty BEZ uczenia maszynowego — tylko przez porownanie geometrii dłoni do wzorcow.

**Czego sie nauczylem:**
- Jak dziala MediaPipe Solutions (21 punktow dłoni)
- Normalizacja i rotacyjna inwariancja wektora punktow
- Automatyczne zbieranie danych z kamery po progu pewnosci

## Jak uruchomic

```powershell
# Aktywuj srodowisko
.\venv\Scripts\Activate.ps1

# Uruchom
python rozpoznawanie_rak.py
```

## Wymagane dane

- `../alfabet/wzor_A.jpg`, `../alfabet/wzor_B.jpg`, ... — jedno zdjecie per gest

## Co robi skrypt

1. Wczytuje wzorce z `../alfabet/` i buduje "baze wiedzy" geometrii
2. Kamery przetwarza na zywo, porownuje szkielet dłoni do kazdego wzorca
3. Jesli pewnosc > 85%, zapisuje klatke do `../do_nauki/` (max raz na 3 sekundy)
4. Wyswietla procenty dopasowania do wszystkich wzorcow na ekranie

## Wniosek

Metoda dziala dla prostych gestow statycznych w kontrolowanych warunkach, ale jest krucha na zmiany kata i odleglosci. Przejscie do etapu 2 (siec neuronowa).
