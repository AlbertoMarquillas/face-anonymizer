import cv2
from face_anonymizer import FaceAnonymizer

#Anonymize faces in a webcam feed
webcam = 0

face_anonymizer = FaceAnonymizer(webcam)

cap = cv2.VideoCapture(webcam)

output_file = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

while cap.isOpened():
    
    ret, frame = cap.read()
    
    
    if not ret:
        break
    
    frame = face_anonymizer.anonymize_faces(draw=True, save=False, frame=frame)
    
    output_file.write(frame)
    
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
output_file.release()