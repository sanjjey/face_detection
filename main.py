import cv2
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
def drawREc(vid):
    gray_image = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
    for x,y,w,h in faces:
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),4)
    return faces
while True:
    result,video_frame = video.read()
    if result is False:
        break
    faces=drawREc(video_frame)
    cv2.imshow("My face detection project" ,video_frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
video.release()
cv2.destroyWindow()
