import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing.DrawingSpec

# Keep original calculate_angle function
def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b)
    c = np.array(c)

    angle = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(angle) * 180.0 / np.pi
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


cap = cv2.VideoCapture(0)

curl_count = 0
curl_stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while 1:
        ret, frame = cap.read()
        
        # Detect pose 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = pose.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(5,221,247), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(5,8,247), thickness=2, circle_radius=2))

        # Get landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Curl counter logic
            if angle > 160:
                curl_stage = "down"
            if angle < 30 and curl_stage =='down':
                curl_stage="up"
                curl_count +=1

        except:
            pass
        
        # Display curl count 
        cv2.rectangle(img, (10,10), (300,80), (0,0,0), -1)
        cv2.putText(img, 'Curls: ' + str(curl_count), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        
        # Render image    
        cv2.imshow('Curl Counter', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
