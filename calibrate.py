import numpy as np
import cv2


cap = cv2.VideoCapture('vtest.avi')
ret, frame = cap.read()

kernel = np.ones((7, 7), np.float32) / 25
blur = cv2.filter2D(frame, -1, kernel)
b,g,r = cv2.split(blur)
#idxhigh= np.logical_or(g>150, b>250)
#idxlow = np.logical_and(r<150,g<150,b<150)
#idx = np.logical_or(idxhigh,r<100)

idxhigh = np.logical_or(g > 150, b > 230)
idxlow = np.logical_and(r < 150, np.logical_and(g < 150, b < 150))
idxwhite = np.logical_and(g > 150, b > 150)

idx = np.logical_or(idxhigh, np.logical_or(r < 100, idxwhite))


test = frame.copy()
test[idx]=0

gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

kernel = np.ones((8, 8), np.uint8)
thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
#edged = cv2.Canny(thresh2, 250, 255)
#cnts, hierachy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh2.copy(), 1, 2)

ctn2 =[]
first =[]
for c in contours:
    if cv2.arcLength(c,True)>280 and cv2.arcLength(c,True)<500:
        #if cv2.isContourConvex(c):
        #approx = cv2.approxPolyDP(c,0.05*cv2.arcLength(c,True),True)
        #print len(approx)
        #if len(approx)==4:
        hull = cv2.convexHull(c, returnPoints=False)
        if np.sum(np.sum(cv2.convexityDefects(c, hull),axis=1), axis=0)[3] < 7800:
            rect=cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(rect)
            box = np.array(box).reshape((-1,1,2)).astype(np.int32)
            #ctn2.append(cv2.convexHull(c, returnPoints=True))
            ctn2.append(c)
            if len(ctn2)==3:
                first.append(c)


cv2.drawContours(frame, first, -1, (255, 0, 0), 3)
cv2.imshow("test", frame)


#Set waitKey
cv2.waitKey()

cap.release()
cv2.destroyAllWindows()