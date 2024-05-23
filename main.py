import cv2
import mediapipe as mp
import time
import numpy as np
from pygame import mixer
from scipy.spatial import distance

def blinked(arr):
    up = distance.euclidean(arr[3], arr[13]) + distance.euclidean(arr[4], arr[12]) + distance.euclidean(arr[5], arr[11])
    down = distance.euclidean(arr[0], arr[8])
    ratio = round((up / (3.0 * down)),2)
    cv2.putText(frame, f'ratio: {ratio}', (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10,25,37), 3)

    if (ratio > 0.27):
        return 2
    elif (ratio > 0.21 and ratio <= 0.26):
        return 1
    else:
        return 0

def yawning(arr):
    up = distance.euclidean(arr[2], arr[3]) + distance.euclidean(arr[6], arr[7]) +distance.euclidean(arr[4], arr[5])
    down = distance.euclidean(arr[0], arr[1])
    y_ratio = up /(3.0*down)

    if (y_ratio < 0.5):
        return 1
    else:
        return 0



cap = cv2.VideoCapture(0)
pTime = 0

mixer.init()
mixer.music.load("beep-warning-6387.mp3")

sleep = 0
drowsy = np.array([0, 0])
active = np.array([0, 0])
status, status_1 = "", ""
color, color_1 = (0, 0, 0), (0, 0, 0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

while True:
    ret, frame = cap.read()
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imageRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec,drawSpec)

            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = frame.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                # print(id,x,y)
                # cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.4, (0, 255, 0), 1)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            all_landmarks = np.array([np.multiply([p.x, p.y], [iw, ih]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            right_eye = all_landmarks[RIGHT_EYE]
            left_eye = all_landmarks[LEFT_EYE]
            mouth = all_landmarks[MOUTH]

            left_blink = blinked(left_eye)
            right_blink = blinked(right_eye)
            yawn_rate = yawning(mouth)

            if (left_blink == 0 or right_blink == 0):
                sleep += 1
                drowsy[0] = 0
                active[0] = 0
                if (sleep > 10):
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)
                    mixer.music.play()

            elif (left_blink == 1 or right_blink == 1):
                sleep = 0
                active[0] = 0
                drowsy[0] += 1
                if (drowsy[0] > 10):
                    status = "Drowsy !"
                    color = (0, 0, 255)

            else:
                drowsy[0] = 0
                sleep = 0
                active[0] += 1
                if (active[0] > 10):
                    status = "Active :)"
                    color = (0, 255, 0)

            if(yawn_rate == 0):
                drowsy[1] += 1
                active[1] = 0
                if(drowsy[1] > 50):
                    status_1 = "Yawning"
                   #cv2.putText(frame, str(drowsy_1), (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    color_1 = (0, 0, 255)
            else:
                active[1] += 1
                drowsy[1] = 0
                if (active[1] > 50):
                    status_1 = ""
                    # cv2.putText(frame, str(active_1), (250, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    # color = (0, 255, 0)


            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, status_1, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_1, 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

