import cv2
import numpy as np
import time

# Load resources
cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('TargetImage1.jpeg')
myVid = cv2.VideoCapture('video.mp4')

if imgTarget is None:
    raise ValueError("Target image not found. Check the file path.")

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)

bf = cv2.BFMatcher()

detection = False
frameCounter = 0
prev_time = time.time()
last_seen_time = time.time()
stable_matrix = None
prev_matrix = None
smoothing_factor = 0.9

while True:
    success, imgWebcam = cap.read()
    if not success:
        break

    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    current_time = time.time()

    matrix = None
    good = []

    if des2 is not None:
        matches = bf.knnMatch(des1, des2, k=2)
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) > 20:
            detection = True
            last_seen_time = current_time

            srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

            if matrix is not None:
                if prev_matrix is None:
                    stable_matrix = matrix
                else:
                    stable_matrix = cv2.addWeighted(prev_matrix, smoothing_factor, matrix, 1 - smoothing_factor, 0)
                prev_matrix = stable_matrix.copy()
        else:
            if current_time - last_seen_time > 2:
                detection = False
                stable_matrix = None
                frameCounter = 0
                myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    else:
        if current_time - last_seen_time > 2:
            detection = False
            stable_matrix = None
            frameCounter = 0
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if detection:
        if frameCounter >= myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        if not success:
            break
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    if stable_matrix is not None and detection:
        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, stable_matrix)

        imgWarp = cv2.warpPerspective(imgVideo, stable_matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)

        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

    # Display FPS
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time
    cv2.putText(imgAug, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display Detection Status
    status_text = "Target Found" if detection else "Target Not Found"
    status_color = (0, 255, 0) if detection else (0, 0, 255)
    cv2.putText(imgAug, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Show result
    cv2.imshow("AR Application", imgAug)
    cv2.waitKey(1)
    frameCounter += 1 if detection else 0
