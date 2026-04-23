@echo off
echo ========================================
echo  Konfiguracja srodowisk wirtualnych
echo  dla wszystkich etapow projektu
echo ========================================
echo.

for /d %%D in (etap_*) do (
    if exist "%%D\requirements.txt" (
        echo [%%D] Tworzenie srodowiska wirtualnego...
        python -m venv "%%D\venv"
        if errorlevel 1 (
            echo [%%D] BLAD: Nie udalo sie utworzyc venv!
        ) else (
            echo [%%D] Instalowanie pakietow z requirements.txt...
            call "%%D\venv\Scripts\activate.bat"
            pip install -r "%%D\requirements.txt"
            call "%%D\venv\Scripts\deactivate.bat"
            echo [%%D] Gotowe!
        )
        echo.
    )
)

echo ========================================
echo  Wszystkie srodowiska zostaly skonfigurowane.
echo  Aby aktywowac srodowisko danego etapu, uzyj:
echo  .\etap_XX_nazwa\venv\Scripts\activate.bat
echo ========================================
pause
