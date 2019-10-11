import cv2
import numpy as np

detector = cv2.CascadeClassifier('templatedata.xml')
camera = cv2.VideoCapture(0)
pics = []
cnt = 0
pics_clicked = 0
name = input("Enter the name:")
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

  if cnt%10==0:
      small = cv2.resize(crop_face, (100, 100))
      pics.append(small)
      pics_clicked+=1
      print("Clicked pics :", pics_clicked)
      if pics_clicked==20:
          break
  cnt+=1
  cv2.imshow("img", im)

pics = np.array(pics)
np.save(name+".npy", pics)
camera.release()