import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Dodatkowo pip install sort
from sort import Sort  # https://github.com/abewley/sort

# --- ARGUMENTY ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file')
parser.add_argument('--source', required=True, help='Image / folder / video / camera')
parser.add_argument('--thresh', default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='WxH for display')
parser.add_argument('--record', action='store_true', help='Record video output')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('Model not found!')
    sys.exit(0)

# --- LOAD MODEL ---
model = YOLO(model_path, task='detect')
labels = model.names

# --- DETERMINE SOURCE TYPE ---
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']


if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list: source_type = 'image'
    elif ext.lower() in vid_ext_list: source_type = 'video'
    else: sys.exit(f'Unsupported file type: {ext}')
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    sys.exit(f'Invalid input: {img_source}')

# --- RESOLUTION ---
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# --- VIDEO / CAMERA CAPTURE ---
if source_type == 'video': cap = cv2.VideoCapture(img_source)
elif source_type == 'usb': cap = cv2.VideoCapture(usb_idx)

if user_res:
    cap.set(3, resW)
    cap.set(4, resH)

# --- RECORDING ---
if record:
    if not user_res:
        print('Please specify resolution for recording.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# --- COLORS ---
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# --- SORT TRACKER ---
tracker = Sort(max_age=15, min_hits=3) # prosty tracker
tracked_objects = {}  # {track_id: classname} -> do "przytrzymania" klasy
tracked_classes = {}

price = {"kinder_bueno":3.62, "knoppers":4.52, "lion":2.71, "price_polo":2.71, "snickers":3.61, "twix":4.98 }
total_amount = 0.0
object_list = []
paragon = []
paragons_counter = 0
day_amount = 0



# --- INFERENCE LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print('End of video / cannot read frame')
        break

    # resize na potrzeby wyświetlania
    frame = cv2.resize(frame, (640,640))

    # YOLO DETECTION
    results = model(frame, imgsz=416, verbose=False)
    detections = results[0].boxes

    # --- PRZYGOTUJ DETECTIONS DLA TRACKERA ---
    # Tracker wymaga listy: [[x1,y1,x2,y2,score], ...]
    dets_for_tracker = []
    for det in detections:
        conf = det.conf.item()
        if conf < min_thresh: continue
        xyxy = det.xyxy.cpu().numpy().squeeze()
        dets_for_tracker.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])

    dets_for_tracker = np.array(dets_for_tracker)

    # --- TRACKING ---
    if len(dets_for_tracker) == 0:
        tracks = tracker.update(np.empty((0,5)))
    else:
        tracks = tracker.update(dets_for_tracker)  # returns [[x1,y1,x2,y2,track_id], ...]

    # --- RYSOWANIE BBOX I PRZYTRZYMANIE KLASY ---

    for trk in tracks:
        x1,y1,x2,y2,track_id = trk
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        track_id = int(track_id)

        # znajdź najlepsze dopasowanie z detekcji
        best_iou = 0
        best_class = None

        for det in detections:
            xyxy = det.xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = map(int, xyxy)

            inter_w = max(0, min(xmax,x2) - max(xmin,x1))
            inter_h = max(0, min(ymax,y2) - max(ymin,y1))
            inter = inter_w * inter_h
            union = (xmax-xmin)*(ymax-ymin) + (x2-x1)*(y2-y1) - inter
            iou = inter / (union + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_class = labels[int(det.cls.item())]

        # jeśli nowy obiekt – zapisz jego klasę
        if track_id not in tracked_classes:
            if best_class is not None:
                tracked_classes[track_id] = best_class
        else:
            # jeśli już był – ignoruj nowe klasy z YOLO
            best_class = tracked_classes[track_id]

        if track_id in tracked_classes:
            label_class = tracked_classes[track_id]

            color = bbox_colors[track_id % 10]
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label = f'{label_class} ID:{track_id}'
            cv2.putText(frame,label,(x1,y1-7),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
            if track_id not in object_list:
                if label_class!="separator":
                    # track_id jeszcze nie dodany
                    paragon.append(label_class)
                    object_list.append(track_id)
                    total_amount += price[label_class]
                else:
                    if total_amount>0:
                        print(f'Paragon: ITEMS: {paragon},TOTAL: {total_amount:.2f}')
                        paragon = []
                        paragons_counter += 1
                        day_amount += total_amount
                        total_amount = 0

    # Display detection results
    cv2.putText(frame, f'Number of objects: {len(object_list)} Total amount: {total_amount:.2f}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO + TRACKING', frame)
    if record: recorder.write(frame)

    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break

print(f'paragonow dzis: {paragons_counter}, obrot razem: {day_amount:.2f}')

cap.release()
if record: recorder.release()
cv2.destroyAllWindows()
