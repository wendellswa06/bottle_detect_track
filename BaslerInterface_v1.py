from pypylon import pylon
import sort
import time
import cv2
import numpy as np
import datetime
from yolo_detect_and_count import YOLOv8_ObjectDetector, YOLOv8_ObjectCounter

# Create an instant camera object with the camera device found first.
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Open camera and set some camera settings
camera.Open()
# camera.GainAuto.SetValue('Off')  # Optional steps to adjust parameters
# camera.Gain.SetValue(10.0)       # Optional steps to adjust parameters
pylon.FeaturePersistence.Load('10_30.pfs', camera.GetNodeMap(), True)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#-----------------------------------------------
yolo_name = "yolov8x.pt"
detector = YOLOv8_ObjectDetector(yolo_name, conf=0.25, iou=0.45, classes=[39])
tracker = sort.Sort(max_age=30, min_hits=10, iou_threshold=0.4)
prev = time.time()
totalCount = []

while camera.IsGrabbing():
    detections = np.empty((0, 5))
    current = time.time()
    interval = current - prev
    prev = current
    print(f"-------------capturing interval = {interval}")
    grabResult = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        image = converter.Convert(grabResult)
        img = image.GetArray()
        img = cv2.resize(img, (1920, 1080))
        conv_img = img[:, 700:1250]
        frame = img.copy()
        frame_width = 1080
        frm_height = 540
        frame = cv2.resize(frame, (frame_width, frm_height))
        frame = cv2.copyMakeBorder(frame, 0, 1080 - frm_height, 0, 1720 - frame_width, cv2.BORDER_CONSTANT, value=(0,0,0))

        beg = time.time()
        # Predict the conveyer image
        results = detector.predict_img(conv_img, verbose=False)
        for box in results.boxes:
            score = box.conf.item() * 100
            class_id = int(box.cls.item())

            x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)
            currentArray = np.array([x1, y1, x2, y2, score])
            detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
        # Get the tracker results
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        # Display current objects IDs
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            id_txt = f"ID:{str(id)}"
            # cv2.putText(img, id_txt, (cx, cy), 4, 1, (0, 0, 255), 2)
            cv2.rectangle(conv_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # if we haven't seen a prticular object ID before, register it in a list 
            if totalCount.count(id) == 0:
                totalCount.append(id)


        if results is None:
            print('***********************************************')
        fps = 1 / (time.time() - beg)
        infer_time = time.time() - beg
        print(f"----------inference time = {infer_time}-----------")

        conv_img = cv2.rotate(conv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_width = conv_img.shape[1]
        cv_height = conv_img.shape[0]
        conv_img = cv2.resize(conv_img, (cv_width * (1040 - frm_height) // cv_height, 1040 - frm_height))
        frame[1100 - frm_height:1060, 20 :cv_width * (1040 - frm_height) // cv_height + 20] = conv_img

        frame = cv2.putText(frame, str(len(totalCount)), (1400,500), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        frame = cv2.putText(frame, "Total Count", (1250, 400), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
        #----------------add time---------------
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        frame = cv2.putText(frame, f"{current_time}", (1250, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        #----------------------------------------

        cv2.namedWindow('Basler Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Basler Camera', frame)

        k = cv2.waitKey(1)
        if k == 27: # if 'ESC' key is pressed
           break

camera.Close()
def bottle_counter(camera):
    while camera.IsGrabbing():
        detections = np.empty((0, 5))
        current = time.time()
        interval = current - prev
        prev = current
        print(f"-------------capturing interval = {interval}")
        grabResult = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data.
            image = converter.Convert(grabResult)
            img = image.GetArray()
            img = cv2.resize(img, (1920, 1080))
            conv_img = img[:, 700:1250]
            frame = img.copy()
            frame_width = 1080
            frm_height = 540
            frame = cv2.resize(frame, (frame_width, frm_height))
            frame = cv2.copyMakeBorder(frame, 0, 1080 - frm_height, 0, 1720 - frame_width, cv2.BORDER_CONSTANT, value=(0,0,0))

            beg = time.time()
            # Predict the conveyer image
            results = detector.predict_img(conv_img, verbose=False)
            for box in results.boxes:
                score = box.conf.item() * 100
                class_id = int(box.cls.item())

                x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)
                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))

            resultsTracker = tracker.update(detections)

            for result in resultsTracker:
            # Get the tracker results
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

            # Display current objects IDs
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                id_txt = f"ID:{str(id)}"
                # cv2.putText(img, id_txt, (cx, cy), 4, 1, (0, 0, 255), 2)
                cv2.rectangle(conv_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # if we haven't seen a prticular object ID before, register it in a list 
                if totalCount.count(id) == 0:
                    totalCount.append(id)


            if results is None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)
            infer_time = time.time() - beg
            print(f"----------inference time = {infer_time}-----------")

            conv_img = cv2.rotate(conv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv_width = conv_img.shape[1]
            cv_height = conv_img.shape[0]
            conv_img = cv2.resize(conv_img, (cv_width * (1040 - frm_height) // cv_height, 1040 - frm_height))
            frame[1100 - frm_height:1060, 20 :cv_width * (1040 - frm_height) // cv_height + 20] = conv_img

            frame = cv2.putText(frame, str(len(totalCount)), (1400,500), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            frame = cv2.putText(frame, "Total Count", (1250, 400), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
            #----------------add time---------------
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            frame = cv2.putText(frame, f"{current_time}", (1250, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            #----------------------------------------

            cv2.namedWindow('Basler Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Basler Camera', frame)

            k = cv2.waitKey(1)
            if k == 27: # if 'ESC' key is pressed
                break

    
if __name__ == "__main__":
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Open camera and set some camera settings
    camera.Open()
    # camera.GainAuto.SetValue('Off')  # Optional steps to adjust parameters
    # camera.Gain.SetValue(10.0)       # Optional steps to adjust parameters
    pylon.FeaturePersistence.Load('10_30.pfs', camera.GetNodeMap(), True)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    #-----------------------------------------------
    yolo_name = "yolov8x.pt"
    detector = YOLOv8_ObjectDetector(yolo_name, conf=0.25, iou=0.45, classes=[39])
    tracker = sort.Sort(max_age=30, min_hits=10, iou_threshold=0.4)
    prev = time.time()
    totalCount = []

    bottle_counter(camera)

    camera.Close()