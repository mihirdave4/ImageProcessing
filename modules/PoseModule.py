import cv2
import mediapipe as mp
import  time

new_width = 1500
new_height = 700
class PoseDetector:
    def __init__(self,mode=False,upBody=False,smooth=True, detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.upbody=upBody
        self.smooth=smooth
        self.detectioncon=detectionCon
        self.trackcon=trackCon


        self.mpdraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.upbody,self.smooth,min_detection_confidence=self.detectioncon,min_tracking_confidence=self.trackcon)

    def findPose(self,img,draw=True):


        landmark_style = self.mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        connection_style = self.mpdraw.DrawingSpec(color=(0, 255, 0), thickness=2)
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpdraw.draw_landmarks(img,results.pose_landmarks,self.mpPose.POSE_CONNECTIONS,landmark_style,connection_style)

        return  img
        # for id ,lm in enumerate(results.pose_landmarks.landmark):
        #     h, w, c=img.shape
        #     cx,cy= int(lm.x*w), int(lm.y*h)



def main():
    cap = cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\jogging.mp4")
    # cap=cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\ashana.mp4")
    # cap=cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\dead_lifting_slowmo.mp4")
    # cap=cv2.VideoCapture(r"C:\Users\mihir\OneDrive\Documents\MotionVideos\breakdance.mp4")
    # cap = cv2.VideoCapture(0)
    pTime=0
    detector=PoseDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (new_width, new_height))
        img=detector.findPose(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()