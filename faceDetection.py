import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()
fpsReader = cvzone.FPS()

print("Press Ctrl-C to terminate while statement")

try:
    while True:
        success, img = cap.read()
        fps, img = fpsReader.update(img,pos=(10,50),color=(0,255,0),scale=2,thickness=2)
        img, bboxs = detector.findFaces(img)

        if bboxs:
            # bboxInfo - "id","bbox","score","center"
            center = bboxs[0]["center"]
            #cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()