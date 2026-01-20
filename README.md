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

python yolo_detect_tracking.py --model my_model.pt --source usb0

python yolo_detect_tracking.py --model my_model.pt --source test_film5.mp4



## Przygotowanie danych, trenowanie, testowanie modelu i uruchomienie lokalne zostało krok po kroku opisane w notatniku Colab:

https://colab.research.google.com/drive/1XDYI_d3Xu1aDj9JxgF9OVOFvgb15RDVH?usp=sharing
 

