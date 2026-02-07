import  cv2
import  numpy as np
import  mediapipe as mp
import  modules.HandTrackingModule as htm
import  time
import  pyautogui

#############################

wcam,hcam=640,480
frameR=100  # Frame Reduction
smoothening=5
#############################


cap=cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
pTime=0
plocx,plocy=0,0
clocx,clocy=0,0
detector=htm.handDetactor(maxhands=1)
wscr,hscr=pyautogui.size()
lmlist=[]
while True:
    # Get the landmarks
    success,img=cap.read()
    img=detector.findHands(img)
    lmlist,bbox=detector.findPosition(img)

    # tip of index and midle finger
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        # check which fingers are up
        fingers=detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR), (255, 0, 255), 2)
        # only index finger moving mode
        if fingers[1]==1 and fingers[2]==0:

            # conver coordinated
            x3=np.interp(x1,(frameR,wcam-frameR),(0,wscr))
            y3 = np.interp(y1, (frameR, hcam-frameR), (0, hscr))
            # smoothen values
            clocx=plocx+(x3-plocx)/smoothening
            clocy=plocy+(y3-plocy)/smoothening
            # move mouse
            pyautogui.moveTo(wscr-clocx,clocy)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocx,plocy=clocx,clocy
        # both index and midle fingers are up:clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # find the distance between fingers
            length,img,infoline=detector.findDistance(8,12,img,draw=False)
            # click mouse if distance short
            if length<40:
                cv2.circle(img,(infoline[4],infoline[5]),15,(255,0,255),cv2.FILLED)
                pyautogui.click()



    # frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    # display

    cv2.imshow('Img',img)
    cv2.waitKey(1)

