import numpy as np
import cv2


cap = cv2.VideoCapture('converted.mp4')
cv2.namedWindow('test',cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 160*4,90*4)
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break


    kernel = np.ones((7, 7), np.float32) / 25
    blur = cv2.filter2D(frame, -1, kernel)

    b,g,r = cv2.split(blur)

    #inv = (blue[0]-(np.ones((len(blue[0]), len(blue[0][0])))*255))*(-1)
    inv = 255 - b


    idxhigh= np.logical_or(g>150, b>230)
    idxlow = np.logical_and(r<150,np.logical_and(g<150,b<150))
    idxwhite = np.logical_and(g>150,b>150)

    idx = np.logical_or(idxhigh,np.logical_or(r<100, idxwhite))
    #idxs = multidim_intersect(np.array(idxr),np.array(idxg))
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
    for c in contours:
        if cv2.arcLength(c,True)>280 and cv2.arcLength(c,True)<500:
            #if cv2.isContourConvex(c):
            #approx = cv2.approxPolyDP(c,0.05*cv2.arcLength(c,True),True)
            #print len(approx)
            #if len(approx)==4:
            hull = cv2.convexHull(c, returnPoints=False)
            #print "bla:", np.sum(np.sum(cv2.convexityDefects(c, hull),axis=1), axis=0)[3]
            if np.sum(np.sum(cv2.convexityDefects(c, hull),axis=1), axis=0)[3] < 7800:
                rect=cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(rect)
                box = np.array(box).reshape((-1,1,2)).astype(np.int32)
                ctn2.append(c)

    cv2.drawContours(frame, ctn2, -1, (255, 0, 0), 3)
    cv2.imshow("test", frame)
    #cv2.imshow("test", thresh2)



    #
    # imCalHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # kernel = np.ones((5, 5), np.float32) / 25
    # blur = cv2.filter2D(imCalHSV, -1, kernel)
    # h, s, imCal = cv2.split(blur)
    # ret, thresh2 = cv2.threshold(imCal, 128, 255, cv2.THRESH_BINARY_INV)
    # kernel = np.ones((3, 3), np.uint8)
    # thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    # #cv2.imshow("thresh2", thresh2)
    # edged = cv2.Canny(thresh2, 250, 255)
    # cv2.imshow("test", edged)



    # cimg = cv2.medianBlur(frame, 5)
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # print "before"
    # circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 800,
    #                            param1=50, param2=30, minRadius=400, maxRadius=0)
    # print "test"
    #
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    #



   # cv2.imshow('frame', cimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()