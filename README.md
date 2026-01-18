# Projekt rozpoznawania produktów na taśmie sklepowej

**Język:** Python 3.9+  
**Biblioteki:** ultralytics, opencv-python, numpy

Projekt wykorzystuje YOLO (Ultralytics) do rozpoznawania produktów na obrazie
i automatycznego podsumowania paragonu.

Źródłem obrazu może być:
- USB kamera (`usb0`, `usb1`...)  
- Plik wideo (`test_film5.mp4`)  
- Pojedyncze zdjęcie (`test.jpg`)  
- Folder z obrazami (`test_dir/`)


## Instalacja

pip install -r requirements.txt


## Propozycja uruchomienia z wybranym źródłem obrazu:

python yolo_detect.py --model my_model.pt --source usb0

python yolo_detect.py --model my_model.pt --source test_film5.mp4

