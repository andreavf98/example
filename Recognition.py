import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


UDP_IP = "127.0.0.1"
UDP_PORT = 5065
last = []
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #cv2.imshow('MediaPipe Hands', image)
        image1=image

        # Change color-space from BGR -> HSV
        hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, np.array([50,100,100]), np.array([80,255,255]))

        # Kernel for morphological transformation    
        kernel = np.ones((15,15))

        # Apply morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask2, kernel, iterations = 2)
        erosion = cv2.erode(dilation, kernel, iterations = 1)    

        # Apply Gaussian Blur and Threshold
        filtered = cv2.GaussianBlur(erosion, (3,3), 0)
        ret,thresh = cv2.threshold(filtered, 127, 255, 0)

        # Show threshold image
        
        #kernel2 = np.ones((10,10))
        #cierre = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
        
        #cv2.imshow("Thresholded", cierre)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

        try:
            # Find contour with maximum area
            contour = max(contours, key = lambda x: cv2.contourArea(x))
            # Create bounding rectangle around the contour
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(image1,(x,y),(x+w,y+h),(0,0,255),0)

            # Find convex hull
            hull = cv2.convexHull(contour)

            # Draw contour
            drawing = np.zeros(image1.shape, np.uint8)
            cv2.drawContours(drawing,[contour],-1,(0,255,0),cv2.FILLED)
            cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
            
            #Centros del contorno
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(drawing, (cX, cY), 7, (255, 255, 255), -1)
            
            # Find convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour,hull)
            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger 
            # tips) for all defects
            count_defects = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(drawing,far,1,[0,0,255],-1)
                    
                cv2.line(drawing,start,end,[0,255,0],2)

            all_image = np.hstack((drawing, image1))
            cv2.imshow('Recognition', all_image)
            
            last.append(count_defects)
            if(len(last) > 5):
                last = last[-5:]
                
            if(count_defects == 0 and 4 in last):
                last = []
                sock.sendto( ("JUMP!").encode(), (UDP_IP, UDP_PORT) )
                print("_"*10, "Jump Action Triggered!", "_"*10)
                print("Coordenada X: ",cX,"    ","Coordenada Y: ",cY)

            if(count_defects == 1 and 4 in last):
                last = []
                sock.sendto( ("FLY!").encode(), (UDP_IP, UDP_PORT) )
                print("_"*10, "FLY action triggered!", "_"*10)

        except:
            pass   


        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()