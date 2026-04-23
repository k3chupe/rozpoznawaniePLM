@echo off
echo ======================================== >> raport_nocny.txt
echo Start programu: %date% %time% >> raport_nocny.txt

echo Uruchamiam program... Czekam na jego zakonczenie...
python .\skrypt_treningowy.py

echo Koniec programu: %date% %time% >> raport_nocny.txt
echo ======================================== >> raport_nocny.txt
echo Program zakonczyl dzialanie. Wyniki zapisano w pliku raport_nocny.txt.
pause