# Uruchom ten skrypt po aktywacji srodowiska, aby zapisac zamrozzone wersje pakietow.
# Dla kazdego etapu: aktywuj venv -> pip freeze -> zapisz do requirements.lock.txt

$etapy = @("etap_04_keras_tuner", "etap_05_lstm_ruch", "etap_02_siec_mlp", "etap_03_xgboost_porownanie")

foreach ($etap in $etapy) {
    $venvPip = Join-Path $PSScriptRoot "$etap\venv\Scripts\pip.exe"
    $lockFile = Join-Path $PSScriptRoot "$etap\requirements.lock.txt"

    if (Test-Path $venvPip) {
        Write-Host "[$etap] Generowanie requirements.lock.txt..."
        & $venvPip freeze | Out-File -FilePath $lockFile -Encoding utf8
        Write-Host "[$etap] Zapisano: $lockFile"
    } else {
        Write-Host "[$etap] Pominiety - brak venv (uruchom najpierw setup_venv.bat)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Gotowe. Pliki requirements.lock.txt zawieraja dokladne wersje pakietow uzyte podczas treningu."
