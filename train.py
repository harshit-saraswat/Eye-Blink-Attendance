import cv2
import face_recognition
import pickle
import os

class TrainImages(object):
    def __init__(self):
        self.TRAIN_PATH = "Train_Images"
        self.ENCODINGS_PATH = "encodings.pkl"

        if not os.path.exists(self.TRAIN_PATH):
            print("WARNING: Train_Images directory not found. Process may fail, create Train_Images Directory to continue.")
            print("-----------------------------------------------------------------------------------------------------------------------------------")
    
    #Function to create face encodings for duplicity
    def create_encodings(self,img):
        rgb_small_frame=img
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if len(face_locations)<1:
            return None
        
        face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
        return face_encoding

    #Function to save encodings as a pickle file
    def save_encodings(self,encs,names):
        data=[]
        d = [{"name": nm, "encoding": enc} for (nm, enc) in zip(names, encs)]
        data.extend(d)
        encodingsFile = self.ENCODINGS_PATH
        
        # dump the facial encodings data to disk
        print("INFO: Serializing Encodings")
        f = open(encodingsFile, "wb")
        f.write(pickle.dumps(data))
        f.close()   
        print("INFO: Training Completed for {} users.".format(len(names))) 

    def train_images(self):
        if not len(os.listdir(self.TRAIN_PATH)):
            print("ERROR: No Images found. Please put images in Train_Images folder.")
            print("-----------------------------------------------------------------------------------------------------------------------------------")
        else:
            images = os.listdir(self.TRAIN_PATH)
            names = []
            face_encodings = []
            for img_name in images:
                image = cv2.imread(self.TRAIN_PATH + "/" + img_name)
                name = img_name.split(".")[0]
                face_encoding = self.create_encodings(image)

                if face_encoding is None:
                    print("INFO: Face image cannot be used for {}. Skipping this user.".format(name))
                else:
                    names.append(name)
                    face_encodings.append(face_encoding)
                    print("INFO: Face image trained for {}.".format(name))
            

            self.save_encodings(face_encodings,names)

if __name__ == "__main__":
    trainObj = TrainImages()
    trainObj.train_images()