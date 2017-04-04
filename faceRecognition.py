import face_recognition
import cv2


class FaceRecognition(object):

    def __init__(self):
        self.userImgs = []

    def setUsersImg(self, userImgs):
        self.userImgs = userImgs

    def findUser(self):

        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            user = 0

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                for img in self.userImgs:
                    user_image = face_recognition.load_image_file(img)
                    user_face_encoding = face_recognition.face_encodings(user_image)[0]

                    isMatch = face_recognition.compare_faces([user_face_encoding], face_encoding)[0]

                    name = ''
                    if isMatch:
                        name = 'user : {}'.format(user)
                        user += 1
                        self.detectSmile(frame)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                break

    def detectSmile(self, frame):

        facePath = "/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
        smilePath = "/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_smile.xml"
        faceCascade = cv2.CascadeClassifier(facePath)
        smileCascade = cv2.CascadeClassifier(smilePath)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scaleFactor = 1.05

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=8,
            minSize=(55, 55),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            detectSmile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=2,
                minNeighbors=20,
                minSize=(25, 25),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            for (x, y, w, h) in detectSmile:
                cv2.rectangle(roi_color, (x, y), (x + w, y + h), (255, 255, 255), 1)


