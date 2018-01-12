import numpy as np
import cv2
import math


cap = cv2.VideoCapture('converted.mp4')
ret, frame = cap.read()

kernel = np.ones((7, 7), np.float32) / 25
blur = cv2.filter2D(frame, -1, kernel)
b,g,r = cv2.split(blur)

idxhigh_r = np.logical_or(g > 150, b > 230)
idxwhite_r = np.logical_and(g > 150, b > 150)

idxhigh_g = np.logical_or(r > 195, b > 255)
idxwhite_g = np.logical_and(r > 180, b > 200)

idx_r = np.logical_or(idxhigh_r, np.logical_or(r < 100, idxwhite_r))
idx_g = np.logical_or(idxhigh_g, np.logical_or(g < 100, idxwhite_g))


find_red = frame.copy()
find_red[idx_r]=0
find_green = frame.copy()
find_green[idx_g]=0

gray_r = cv2.cvtColor(find_red, cv2.COLOR_BGR2GRAY)
_, thresh_r = cv2.threshold(gray_r, 5, 255, cv2.THRESH_BINARY)
gray_g = cv2.cvtColor(find_green, cv2.COLOR_BGR2GRAY)
_, thresh_g = cv2.threshold(gray_g, 5, 255, cv2.THRESH_BINARY)

kernel = np.ones((8, 8), np.uint8)
thresh_r = cv2.morphologyEx(thresh_r, cv2.MORPH_OPEN, kernel)
thresh_r = cv2.morphologyEx(thresh_r, cv2.MORPH_CLOSE, kernel)
thresh_g = cv2.morphologyEx(thresh_g, cv2.MORPH_OPEN, kernel)
thresh_g = cv2.morphologyEx(thresh_g, cv2.MORPH_CLOSE, kernel)
contours_r, _ = cv2.findContours(thresh_r.copy(), 1, 2)
contours_g, _ = cv2.findContours(thresh_g.copy(), 1, 2)

ctn2 =[]
first =[]

contours=np.append(contours_r,contours_g)

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


empty = np.zeros((len(b), len(b[0])), np.uint8)
cv2.drawContours(empty, ctn2, -1, (255, 0, 0), 3)



_, empty = cv2.threshold(empty, 5, 255, cv2.THRESH_BINARY)

kernel = np.ones((48, 48), np.uint8)
empty = cv2.morphologyEx(empty, cv2.MORPH_CLOSE, kernel)

contours_ring, _ = cv2.findContours(empty.copy(), 1, 2)
#cv2.drawContours(frame, contours_ring, -1, (0, 0, 255), 3)
outsideE = None
minX = 9999999999999

for cnt in contours_ring:
    c = cnt
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    if extLeft[0]<minX:
        minX=extLeft[0]

    else:
        continue

    #cv2.circle(frame, extLeft, 8, (255,0,0), -1)
    #cv2.circle(frame, extRight, 8, (255,0,0), -1)
    #cv2.circle(frame, extTop, 8, (255,0,0), -1)
    #cv2.circle(frame, extBot, 8, (255,0,0), -1)

    #ellipse = cv2.fitEllipse(np.array([extLeft,extRight,extTop,extBot,extBot]))
    outsideE = cv2.fitEllipse(cnt)
    #cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

    #x, y = ellipse[0]
    #a, b = ellipse[1]
    #angle = ellipse[2]

    #center_ellipse = (x, y)

    #a = a / 2
    #b = b / 2

    #cv2.ellipse(frame, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0,
    #            cv2.cv.CV_RGB(255, 0, 0))


cv2.ellipse(frame, outsideE, (255, 0, 0), 2)

cv2.imshow("test", frame)


#Set waitKey
cv2.waitKey()

cap.release()
cv2.destroyAllWindows()