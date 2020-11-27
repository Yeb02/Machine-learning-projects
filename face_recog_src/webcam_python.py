import sys
import cv2
import pygame

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vc = cv2.VideoCapture(0)
vc.set(3, 1280)
vc.set(4, 720)

pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("help me")
pygame.event.get()
#720p


if vc.isOpened(): # try to get the first frame
    rval, img = vc.read()
else:
    rval = False

a = 0
while rval:
    pygame.event.get()
    keys = pygame.key.get_pressed()
    rval, img = vc.read()
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if keys[pygame.K_s]:
        filename = '/home/edern/Documents/TIPE/traitement/mesures/test_image_%06d.jpg' % a
        cv2.imwrite(r'' + filename, img)
        print(a)
        a += 1
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("preview", img)
    key = cv2.waitKey(20)

    if key == 27: #exit on ESC
        break
    """if keyboard.is_pressed('q'):
        break"""
cv2.destroyWindow("preview")
vc.release()
pygame.quit()
