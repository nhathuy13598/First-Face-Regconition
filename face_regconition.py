import cv2
face_xml = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_xml = cv2.CascadeClassifier("haarcascade_eye.xml")
def face_regcognition(gray,original):
    face = face_xml.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(original, (x,y), (x+w,y+h),(255,0,0),2)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = original[y:y+h, x:x+w]
        eye = eye_xml.detectMultiScale(eye_gray,1.1,3)
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(eye_color,(ex, ey),(ex+ew, ey+eh),(0,255,0),2)
    return original

video = cv2.VideoCapture('demo.mp4')
while True:
    _,frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    canvas = face_regcognition(gray,frame)
    cv2.imshow("Video",canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()