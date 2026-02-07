import cv2
import mediapipe as mp
import  time

# cap=cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\jogging.mp4")
# cap=cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\ashana.mp4")
# cap=cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\dead_lifting_slowmo.mp4")
# cap=cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\breakdance.mp4")
cap=cv2.VideoCapture(0)
new_width = 1500
new_height = 700
pTime=0
mpdraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()
landmark_style = mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
connection_style = mpdraw.DrawingSpec(color=(0, 255, 0), thickness=2)
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if results.pose_landmarks:
        mpdraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS,landmark_style,connection_style)
        for id ,lm in enumerate(results.pose_landmarks.landmark):
            h, w, c=img.shape
            cx,cy= int(lm.x*w), int(lm.y*h)

    img = cv2.resize(img, (new_width, new_height))
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break