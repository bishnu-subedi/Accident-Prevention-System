"""Collecting Images of eyes and faces of people"""

import cv2
size = 2
cap = cv2.VideoCapture(0) #Use camera 0

#loading the xml file(Model)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


i = 0
j = 0

while True:
    (rval, im) = cap.read()
    #im=cv2.flip(im,1,0) #flip

    # Resizing the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # Detecting MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)

        
        sub_face = im[y+30:y+h, x:x+60]   #right
        sub_face1 = im[y+30:y+h, x+180:x+w]   #x+140:x+w

        """The detected facial image is splitted into two different images.
           Collecting image of right and left portion of every facial images.
           These images are used to train if there is presence of
           cellphone/mobilephone. This is done to cut the training time on
           laptop."""
        FaceFileName = "faceobj/face_" + str(i) + ".png"
        i = i+1
        FaceFileName1 = "faceobj/face_" + str(i) + ".png"
        cv2.imwrite(FaceFileName, sub_face)
        cv2.imwrite(FaceFileName1, sub_face1)
                


        # Detecting the eyes within face
        eyes = eye_cascade.detectMultiScale(mini)

        #Drawing rectangle around the eyes
        for e in eyes:
            (ex, ey, ew, eh) = [ve * size for ve in e] #Scale the shapesize backup
            cv2.rectangle(im, (ex, ey), (ex + ew, ey + eh),(255,0,0),thickness=1)
                        
            sub_face2 = im[ey:ey+eh, ex:ex+ew]
            FaceFileName2 = "eyeTrain/open_" + str(j) + ".png"
            cv2.imwrite(FaceFileName2, sub_face2)
                        

    # Show the image
    cv2.imshow('img',   im)


    """If 'q' is pressed then break out of the loop. End the program."""
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

    i = i+1
    j = j+1


cap.release()
cv2.destroyAllWindows()
    
