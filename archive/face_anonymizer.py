import cv2
import mediapipe as mp
import os
import argparse

# Path to the image
image_path = 'Face_anonymizer\data\image3.jpg'

#read the image
image = cv2.imread(image_path)

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', help='Mode of operation: image or video', default='v')
parser.add_argument('-i', '--image', help='Path to the image', default='image')
parser.add_argument('-d', '--draw', type=bool, default=True, help='Draw the detected faces')
parser.add_argument('-s', '--save', type=bool, default=False, help='Save the image')

args = parser.parse_args()

#args -> Namespace(mode='image', image='image', draw=True, save=False)

#Now we're creating a class with all the above code



class FaceAnonymizer:
    def __init__(self, image_path=None, webcam=None):
        if args.mode in ['image', 'i']:
            self.image_path = image_path
            self.image = cv2.imread(image_path)
        if args.mode in ['video', 'v']:
            self.webcam = webcam
            self.image = None
        
    def process_image(self, imageRGB, model):

        H, W, _ = self.image.shape
            
        imageRGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        result = model.process(imageRGB)
        
        if result.detections is not None:
            for detection in result.detections:
                bboxC = detection.location_data.relative_bounding_box

                x1, y1, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)
                
                                    
                if args.draw:
                    self.image = cv2.rectangle(self.image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                    
        return self.image
                    
    def blur_image(self, imageRGB, model):
        H, W, _ = self.image.shape
        
        imageRGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        result = model.process(imageRGB)
        
        if result.detections is not None:
            for detection in result.detections:
                bboxC = detection.location_data.relative_bounding_box

                x1, y1, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                self.image[y1:y1+h, x1:x1+w] = cv2.blur(self.image[y1:y1+h, x1:x1+w], (25, 25))   
                         
        return self.image
        
    def detect_faces(self, draw=True, save=False, frame=None):
        mpFaceDetection = mp.solutions.face_detection
        
        with mpFaceDetection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            
            if args.mode in ['image', 'i']:
                
                self.process_image(self.image, face_detection)
                          
                if args.save:
                    cv2.imwrite(os.path.join('Face_anonymizer\data', 'detected_faces.jpg'), self.image)
                    
            if args.mode in ['video', 'v']:
                
                if frame is not None:
                    
                    self.image = frame
                    
                self.image = self.process_image(self.image, face_detection)
                    
                    
        return self.image
            
    def anonymize_faces(self, draw=True, save=False, frame=None):
        mpFaceDetection = mp.solutions.face_detection
        
        with mpFaceDetection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

                    
            if args.mode in ['image', 'i']:
                
                self.blur_image(self.image, face_detection)
                          
                if args.save:
                    cv2.imwrite(os.path.join('Face_anonymizer\data', 'detected_faces.jpg'), self.image)
                    
            if args.mode in ['video', 'v']:
                
                if frame is not None:
                    
                    self.image = frame
                    
                self.image = self.blur_image(self.image, face_detection)
                    
                    
        return self.image
