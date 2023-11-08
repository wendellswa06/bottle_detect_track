import shutil
import os

import numpy as np
import random
import cv2
import sort
import time

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from yolo_detect_and_count import YOLOv8_ObjectDetector, YOLOv8_ObjectCounter

vid_results_path = '.\\video_object_detection_results'
test_vids_path = '.\\test vids'

######################################## Testing with Images########################################

# img = cv2.imread('.\\test imgs\\bottles.jpg')
# img = cv2.resize(img, (1920, 1080))
# img = np.asarray(img)
# yolo_names = ['yolov8x.pt']
# colors = []
# for _ in range(80):
#     rand_tuple = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
#     colors.append(rand_tuple)

# counters = []
# for yolo_name in yolo_names:
#     counter = YOLOv8_ObjectCounter(yolo_name, conf = 0.10, classes=[39] )
#     counters.append(counter)

# for idx, counter in enumerate(counters):
#     results = counter.predict_img(img)
    
#     for result in results:
#         lt_point = (int(result.boxes.cpu().numpy().data[0][0]), int(result.boxes.cpu().numpy().data[0][1]))
#         br_point = (int(result.boxes.cpu().numpy().data[0][2]), int(result.boxes.cpu().numpy().data[0][3]))
#         cv2.rectangle(img, lt_point, br_point, colors[idx], 2)

#         cv2.putText(img, "%s : %f"%(result.names[int(result.boxes.cls.cpu().numpy()[0])], result.boxes.conf.cpu().numpy()), 
#                     lt_point, cv2.FONT_HERSHEY_PLAIN, 1, colors[idx], 2)

# cv2.imshow('Output', img)
# cv2.waitKey(0)


######################################## Testing On Videos ########################################

video_path = 'test vids\\bottle2.mp4'
out_path = '.\\result vids'
counter = YOLOv8_ObjectCounter('yolov8x.pt', conf=0.2,  iou=0.45, classes=[39],
                               track_max_age=45, track_min_hits=15, track_iou_threshold=0.3)

cap = cv2.VideoCapture(video_path)
vid_name = os.path.basename(video_path)

width = int(cap.get(3))
height = int(cap.get(4))

if not os.path.isdir(out_path):
    os.makedirs(out_path)

save_name = 'yolov8x--' + vid_name
save_file = os.path.join(out_path, save_name.split('.')[0] + '.avi')

out_writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width//2, height//2))

if not cap.isOpened():
    print("Error opening video stream or file")

tracker = sort.Sort(max_age=30, min_hits=10, iou_threshold=0.4)
totalCount = []
currentArray = np.empty((0, 5))

# Read the video frames
frame_count = 0
while cap.isOpened():
    detections = np.empty((0, 5))
    ret, frame = cap.read()

    if not ret:
        print("Error reading frames")
        break

    frame = cv2.resize(frame, (width//2, height//2))
    conveyer_img = frame[:, 700:1250]

    starttime = time.time()
    results = counter.predict_img(conveyer_img, verbose=False)
    if results == None:
        print("************************")
    
    for box in results.boxes:
        score = box.conf.item() * 100
        class_id = int(box.cls.item())

        x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)
        currentArray = np.array([x1, y1, x2, y2, score])
        detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
    #print(type(result))

        # Get the tracker results
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        #print(result)

        # Display current objects IDs
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        id_txt = f"ID: {str(id)}"
        cv2.putText(frame, id_txt, (cx+700, cy), 4, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1+700, y1), (x2+700, y2), (255, 0, 0), 2)

        # if we haven't seen a prticular object ID before, register it in a list 
        if totalCount.count(id) == 0:
            totalCount.append(id)

    # Display Counting results
    count_txt = f"TOTAL COUNT : {len(totalCount)}"
    frame = cv2.putText(frame, count_txt, (5,45), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    # Display FPS on frame
    fps = 1 / (time.time() - starttime)
    frame = cv2.putText(frame,f"FPS : {fps:,.2f}" , 
                        (5,70), cv2.FONT_HERSHEY_COMPLEX, 
                    1,  (0,255,255), 2, cv2.LINE_AA)

    frame_count+=1
    frame = cv2.putText(frame,f"Frames : {frame_count}" , 
                        (5,95), cv2.FONT_HERSHEY_COMPLEX, 
                    1,  (0,255,255), 2, cv2.LINE_AA)

    frame = cv2.line(frame, (700, 0), (700, height//2), (0, 0, 255), 2)
    frame = cv2.line(frame, (1250, 0), (1250, height//2), (0, 0, 255), 2)
    out_writer.write(frame)
    # cv2.imwrite(save_file[:-4] + '-%04d.jpg'%frame_count, frame)
    cv2.imshow('out', frame)
    # cropped_frame = frame[:, 700:1250]
    # cv2.imshow('cropped', cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
