import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')


EYE_AR_THRESH = 1
MOUTH_AR_THRESH = 0.45
BLINK_DURATION_THRESH = 0.2  # Time in seconds for a blink
MOUTH_MOVEMENT_DURATION_THRESH = 1.0  # Time in seconds for mouth movements

#  (MAR)
def calculate_mouth_aspect_ratio(mouth):
    mouth_width = mouth[1][0] - mouth[0][0]
    mouth_height = mouth[3][1] - mouth[1][1]
    aspect_ratio = mouth_height / mouth_width
    return aspect_ratio

# (EAR)
def calculate_eye_aspect_ratio(eye):
    eye_width = eye[1][0] - eye[0][0]
    eye_height = eye[2][1] - eye[0][1]
    aspect_ratio = eye_height / eye_width
    return aspect_ratio

# Blink & Mouth Detection
last_blink_time = time.time()
last_mouth_move_time = time.time()

blink_rate = 0
mouth_rate = 0

def detect_blink_and_mouth_movement(frame, gray):
    global last_blink_time, last_mouth_move_time
    global blink_rate, mouth_rate

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        eye_blink_detected = False

        for (ex, ey, ew, eh) in eyes:
            eye = [(ex, ey), (ex + ew, ey), (ex, ey + eh), (ex + ew, ey + eh)]
            ear = calculate_eye_aspect_ratio(eye)
            print("EAR : ", ear)

            
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Detect blink based on EAR
            if ear >= EYE_AR_THRESH:
                current_time = time.time()
                # if current_time - last_blink_time > BLINK_DURATION_THRESH:
                last_blink_time = current_time
                eye_blink_detected = True
                blink_rate = blink_rate + 1
                print("Blink Detected")
                cv2.putText(frame, "Blink Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detect mouths
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 20)
        mouth_open_detected = False

        for (mx, my, mw, mh) in mouths:
            
            if y + my > y + h // 2:
                mouth = [(mx, my), (mx + mw, my), (mx, my + mh), (mx + mw, my + mh)]
                mar = calculate_mouth_aspect_ratio(mouth)
                print("MAR : ", mar)
                # Draw rectangle around the mouth
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)

                # Detect mouth open based on MAR
                if mar > MOUTH_AR_THRESH:
                    current_time = time.time()
                    
                    last_mouth_move_time = current_time
                    mouth_open_detected = True
                    mouth_rate = mouth_rate + 1
                    print("Mouth Movement Detected")
                    cv2.putText(frame, "Mouth Open Detected", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Fake detection logic
        current_time = time.time()
        time_since_last_blink = current_time - last_blink_time
        time_since_last_mouth_move = current_time - last_mouth_move_time

       

        # If no blinks or mouth movements for 10 seconds, suspect spoofing
        print(f"Mouth : {mouth_rate} , Blink : {blink_rate}")
        if blink_rate > 8 and mouth_rate > 5:
            cv2.putText(frame, "INFO : Real", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "WARNING : Spoof", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    frame = detect_blink_and_mouth_movement(frame, gray)

    
    cv2.putText(frame, "Say your Name ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Eye Blink and Mouth Open/Close Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
