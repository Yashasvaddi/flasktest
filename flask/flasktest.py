from flask import Flask, Response,render_template
import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import math as m

app = Flask(__name__)

# Open the default webcam (use 0 for default, 1 for external camera)
camera = cv2.VideoCapture(0)

def generate_frames():
    mp_hands=mp.solutions.hands
    hands=mp_hands.Hands()
    mp_draw=mp.solutions.drawing_utils  
    
    while True:
        success, frame = camera.read()  # Read frame from webcam
        if not success:
            break  # Stop if failed to read frame
        frame=cv2.flip(frame,1)
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                    h,w,_=frame.shape
                    mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS) 
                    thumb=hand_landmarks.landmark[4]
                    index=hand_landmarks.landmark[8]
                    x1,y1=int(thumb.x*w),int(thumb.y*h)
                    x2,y2=int(index.x*w),int(index.y*h)
                    rect1x1,rect1y1=int(10),int(20)
                    rect1x2,rect1y2=int(120),int(65)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),2)
                    distance=int((m.sqrt((x2-x1)**2+(y2-y1)**2)))
                    if ((rect1x1<int((x1+x2)/2)<rect1x2 and rect1y1<int((y1+y2)/2)<rect1y2)and(int(distance)<=50)):
                        cv2.circle(frame,(int((x1+x2)/2),int((y1+y2)/2)),7,(0,0,0),5)
                        cv2.circle(frame,(int((x1+x2)/2),int((y1+y2)/2)),9,(100,100,100),5)
                        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)
                        frame[:]=255
                        bpoints = [deque(maxlen=1024)]
                        gpoints = [deque(maxlen=1024)]
                        rpoints = [deque(maxlen=1024)]
                        ypoints = [deque(maxlen=1024)]
                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0
                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
                        colorIndex = 0
                        while(True):
                            # frame=np.zeros((471, 636, 3)) + 255
                            ret,frame=camera.read()
                            # if (camflag):
                            frame=cv2.flip(frame,1)
                            result=hands.process(rgb_frame)
                            rect1x1,rect1y1=int(10),int(410)
                            rect1x2,rect1y2=int(120),int(440)
                            cv2.rectangle(frame,(rect1x1,rect1y1),(rect1x2,rect1y2),(0,0,0),2)
                            cv2.putText(frame,"QUIT",(35,435),cv2.FONT_HERSHEY_TRIPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
                            cv2.rectangle(frame,(rect1x1+480,rect1y1),(rect1x2+510,rect1y2),(0,0,0),2)
                            cv2.putText(frame,"SAVE",(525,435),cv2.FONT_HERSHEY_TRIPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
                            if (((rect1x1+480<int((x1+x2)/2)<rect1x2+510 and rect1y1<int((y1+y2)/2)<rect1y2)and(int(distance)<=50))):
                                if(flag):
                                    cv2.putText(frame,"IMAGE SAVED",(255,435),cv2.FONT_HERSHEY_TRIPLEX,1.5,(0,0,0),1,cv2.LINE_AA)
                                #   save_canvas(frame)  
                                flag=False
                            else:
                                flag=True

                            #AIR CANVAS#

                            if result.multi_hand_landmarks:
                                for hand_landmarks in result.multi_hand_landmarks:
                                    # mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                                    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                                    h,w,_=frame.shape
                                    thumb=hand_landmarks.landmark[4]
                                    index=hand_landmarks.landmark[8]
                                    x1,y1=int(thumb.x*w),int(thumb.y*h)
                                    x2,y2=int(index.x*w),int(index.y*h)
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),2)
                                    distance=int((m.sqrt((x2-x1)**2+(y2-y1)**2)))
                                    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)
                                    cv2.circle(frame,(int((x1+x2)/2),int((y1+y2)/2)),7,(0,0,0),5)
                                    cv2.circle(frame,(int((x1+x2)/2),int((y1+y2)/2)),9,(100,100,100),5) 
                                    cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
                                    cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
                                    cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
                                    cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
                                    cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
                                    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                                    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                                    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                                    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                                    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                                    if result.multi_hand_landmarks:
                                        landmarks = []
                                        for handslms in result.multi_hand_landmarks:
                                            for lm in handslms.landmark:
                                                lmx = int(lm.x * 640)
                                                lmy = int(lm.y * 480)
                                                landmarks.append([lmx, lmy])
                                            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
                                            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

                                        fore_finger = (landmarks[8][0], landmarks[8][1])
                                        center1 = fore_finger
                                        thumb = (landmarks[4][0], landmarks[4][1])
                                        center=(int((center1[0]+thumb[0])/2),int((center1[1]+thumb[1])/2))
                                        distance=int(m.sqrt((center1[0]-thumb[0])**2+(center1[1]-thumb[1])**2))
                                        if(distance>110):
                                            cv2.line(frame,thumb,center1,(0,0,255),3)
                                            cv2.circle(frame, center, 3, (0, 255, 0), -1)
                                            cv2.line(frame,thumb,center1,(0,0,255),3)
                                            cv2.circle(frame, center, 5, (0,0,255), -1)
                                        else:
                                            cv2.line(frame,thumb,center1,(0,255,0),3)
                                            cv2.circle(frame, center, 3, (255, 0, 255), -1)
                                            cv2.line(frame,thumb,center1,(0,255,0),3)
                                            cv2.circle(frame, center, 5, (255, 0, 255), -1)

                                        if (thumb[1] - center[1] < 30):
                                            bpoints.append(deque(maxlen=512))
                                            blue_index += 1
                                            gpoints.append(deque(maxlen=512))
                                            green_index += 1
                                            rpoints.append(deque(maxlen=512))
                                            red_index += 1
                                            ypoints.append(deque(maxlen=512))
                                            yellow_index += 1

                                        elif center[1] <= 65:
                                            if 40 <= center[0] <= 140:
                                                bpoints = [deque(maxlen=512)]
                                                gpoints = [deque(maxlen=512)]
                                                rpoints = [deque(maxlen=512)]
                                                ypoints = [deque(maxlen=512)]
                                                blue_index = 0
                                                green_index = 0
                                                red_index = 0
                                                yellow_index = 0
                                                frame[67:, :, :] = 255
                                            elif 160 <= center[0] <= 255:
                                                colorIndex = 0
                                            elif 275 <= center[0] <= 370:
                                                colorIndex = 1
                                            elif 390 <= center[0] <= 485:
                                                colorIndex = 2
                                            elif 505 <= center[0] <= 600:
                                                colorIndex = 3
                                        else:
                                            if colorIndex == 0:
                                                bpoints[blue_index].appendleft(center)
                                            elif colorIndex == 1:
                                                gpoints[green_index].appendleft(center)
                                            elif colorIndex == 2:
                                                rpoints[red_index].appendleft(center)
                                            elif colorIndex == 3:
                                                ypoints[yellow_index].appendleft(center)

                                    points = [bpoints, gpoints, rpoints, ypoints]
                            for i in range(len(points)):
                                for j in range(len(points[i])):
                                    for k in range(1, len(points[i][j])):
                                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                                            continue
                                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as multipart content
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
