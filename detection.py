import os
import numpy as np
import cv2
import bundle

b1 = []
b2 = []
cnt=0
d = {}
for f in os.listdir():
    if f.endswith(".npy"):
        imgs = np.load(f)
        b1.append(imgs)
        labels = np.ones(imgs.shape[0])*cnt
        b2.append(labels)
        d[cnt] = f[:-4]
        cnt+=1

X = np.array(b1)
Y = np.array(b2)
X = X.reshape((20*cnt, 100, 100, 3))
Y = Y.reshape((20*cnt,))


detector = cv2.CascadeClassifier('templatedata.xml')
camera = cv2.VideoCapture(0)

while True:
  b,im = camera.read()

  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'):
      break
  if b ==False:
    continue

  a = im.copy()
  faces = detector.detectMultiScale(im, 1.3)
  if (len(faces)==0):
      cv2.imshow("img", im)
      continue

  f = faces[0]

  x, y, w, h = f
  green = (0, 255, 0)
  cv2.rectangle(im, (x, y), (x + w, y + h), green, 5)
  crop_face = im[y:y+h, x:x+h]

  small = cv2.resize(crop_face, (100, 100))
  tx = d[bundle.knn(X, Y, small)]
  cv2.putText(im, tx, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
  cv2.imshow("img", im)
camera.release()