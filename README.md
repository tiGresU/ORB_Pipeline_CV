import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
# Option 1: Acquire from webcam (real-time)
cap = cv2.VideoCapture(0)  # 0 = default camera
ret, frame = cap.read()    # Capture a frame
if ret:
    cv2.imwrite('acquired_image.jpg', frame)  # Save for next steps
cap.release()

import cv2
img = cv2.imread('acquired_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
blur = cv2.GaussianBlur(gray, (5,5), 0)       # Denoise
equalized = cv2.equalizeHist(blur)            # Contrast enhancement
cv2.imshow('Preprocessed', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('preprocessed.jpg', equalized)

import cv2
img = cv2.imread('preprocessed.jpg', 0)
_, thresh = cv2.threshold(img, 127, 250, cv2.THRESH_BINARY)  # Binary segmentation
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, contours, -1, (0,255,0), 2)  # Draw segmented regions
cv2.imshow('Segmented', output)
cv2.waitKey(0)

import cv2
img = cv2.imread('segmented.jpg', 0)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)  # Extract features
output = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))
cv2.imshow('Features', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('segmented.jpg', 0)
img2 = cv2.imread('another_image.jpg', 0)  # existing image

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

output = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:20],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12,6))
plt.imshow(output, cmap='gray')
plt.axis('off')
plt.show()

import cv2
img = cv2.imread('segmented.jpg', 0)
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    area = cv2.contourArea(cnt)
    label = "Large" if area > 1000 else "Small"  # Simple classification
    cv2.drawContours(output, [cnt], 0, (0,255,0), 2)
    cv2.putText(output, label, (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
cv2.imshow('Classified', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
# Assume detections: list of [x,y,w,h,score]
detections = np.array([[100,100,50,50,0.9], [105,105,50,50,0.8]])  # Example
indices = cv2.dnn.NMSBoxes(detections[:,:4].tolist(), detections[:,4].tolist(), 0.5, 0.4)
img = cv2.imread('segmented.jpg')
for i in indices:
    x,y,w,h = detections[i,:4].astype(int)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imshow('Post-processed Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyAllWindows()
cv2.imwrite('segmented.jpg', output)
