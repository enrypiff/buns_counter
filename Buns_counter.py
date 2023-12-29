import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# define the video to read, because I don't have a webcam on the machine
cap = cv2.VideoCapture("C:/Users/enric/Downloads/video_bun (15).mp4")

# define the model to use. It was trained on a custom dataset annotated by myself
model = YOLO("C:/Users/enric/Downloads/last (2).pt")

class_names = ["bun"]

mask = cv2.imread("mask1.png")

# tracking
tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.4)

limits = [0, 590, 380, 450]
total_count = []

# Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define the codec and filename
out = cv2.VideoWriter('output15.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15.0, (frame_width,frame_height))

prev_frame_time = 0
new_frame_time = 0

while (cap.isOpened()):
    new_frame_time = time.time()
    success, img = cap.read()
    if success:
        img_region = cv2.bitwise_and(img, mask)

        img_graphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
        # use cvzone because it's easier to overlap two images
        img = cvzone.overlayPNG(img, img_graphics, (0, 0))
        results = model(img_region, stream=True)



        # remove the logo from the video
        roi = img[0:70, 620:720]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.blur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        img[0:0 + roi.shape[0], 620:620 + roi.shape[1]] = roi

        roi = img[340:430, 430:560]
        # applying a blur over this new rectangle area
        roi = cv2.blur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        img[340:340 + roi.shape[0], 430:430 + roi.shape[1]] = roi


        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                current_class = class_names[cls]

                if current_class == "bun" and conf > 0.4:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        results_tracker = tracker.update(detections)

        # show the line on which the counting is made
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 255), 5)

        for result in results_tracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

            # the line for the counting is inclined so we need to use the formula to get the line y=mx + q
            m = (limits[3] - limits[1]) / (limits[2] - limits[0])
            q = ((limits[2]*limits[1]) - (limits[0]*limits[3]))/(limits[2] - limits[0])

            yr = cx * m + q

            if limits[0] < cx < limits[2] and yr - 40 < cy < yr + 5:
                if total_count.count(id) == 0:
                    total_count.append(id)
                    # change color of the line and the circle when the counting is made
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # write the total counting
        cv2.putText(img, str(len(total_count)), (80, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 2)
        # calculate the frame rate of the processed video
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        # show on screen the frame
        cv2.imshow("Image", img)
        # write the  frame in the video
        out.write(img)

        cv2.waitKey(1)
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
