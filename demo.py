from faceRecognotion import FaceRecognition


if __name__ == '__main__':
    face = FaceRecognition()
    face.setUsersImg(['266.jpg', '87.jpg'])
    face.findUser()
