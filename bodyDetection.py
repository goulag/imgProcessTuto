import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()
cap = cv2.VideoCapture(0)
fpsReader = cvzone.FPS()

print("Press Ctrl-C to terminate while statement")

try:
    while True:
        success, img = cap.read()
        fps, img = fpsReader.update(img,pos=(10,50),color=(0,255,0),scale=2,thickness=2)

        img = detector.findPose(img)
        lmlist, bbox = detector.findPosition(img, bboxWithHands=True)
        cv2.imshow("MyBodyPoseDetector", img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
