from pypylon import pylon
import numpy as np
import cv2
import time

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    # Open camera and set some camera settings
camera.Open()
# camera.GainAuto.SetValue('Off')  # Optional steps to adjust parameters
# camera.Gain.SetValue(10.0)       # Optional steps to adjust parameters
pylon.FeaturePersistence.Load('Basler_config/11_8.pfs', camera.GetNodeMap(), True)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()


# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
prev = time.time()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (camera.Width.GetValue(), camera.Height.GetValue()))
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080))

while camera.IsGrabbing():
    
    current = time.time()
    interval = current - prev
    prev = current
    print(f"-------------capturing interval = {interval}")
    grabResult = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        image = converter.Convert(grabResult)
        img = image.GetArray()
        frame = img.copy()
        # frame_width = 1080
        # frm_height = 540
        frame = cv2.resize(frame, (1920, 1080))
        out.write(frame)
        
        
        cv2.namedWindow('Basler Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Basler Camera', frame)

        k = cv2.waitKey(1)
        if k == 27: # if 'ESC' key is pressed
            break
