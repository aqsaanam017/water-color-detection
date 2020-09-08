# importing modules

from cv2 import cv2
import numpy as np

# capturing video through webcam
cap = cv2.VideoCapture("test6.mp4")

while(1):
    _, img = cap.read()

    # converting frame(img i.e BGR) to HSV (hue-saturation-value)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brown_lower = np.array([15, 15, 80], np.uint8)
    brown_upper = np.array([24, 200, 200], np.uint8)

    brown = cv2.inRange(hsv, brown_lower, brown_upper)
  

    # Morphological transformation, Dilation
    kernal = np.ones((5, 5), "uint8")

    brown = cv2.dilate(brown, kernal)
    res = cv2.bitwise_and(img, img, mask=brown)

    # Tracking the brown Color
    contours,hierachy=cv2.findContours(brown,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):

            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, "brown color", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255))

    # cv2.imshow("brown",brown)
    cv2.imshow("Color Tracking", img)
    # cv2.imshow("brown",res)
    if cv2.waitKey(23) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
