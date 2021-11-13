import cv2
import face_recognition
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial import distance as dist

class Attendance(object):
    def __init__(self):
        self.ENCODINGS_PATH = "encodings.pkl"
        self.ATTENDANCE_FILE = "./Attendance.xlsx"
        self.EYE_AR_THRESH = 0.25
        self.ATTENDANCE_DATA = pd.read_excel(self.ATTENDANCE_FILE)

        if not os.path.exists(self.ENCODINGS_PATH):
            self.ENCODINGS,self.NAMES = None, None
            print("WARNING: Encodings File not found. Process may fail, please run the train script first.")
            print("-----------------------------------------------------------------------------------------------------------------------------------")
        else:
            self.ENCODINGS,self.NAMES = self.read_encodings()

    #Function to read encodings
    def read_encodings(self):
        data = pickle.loads(open(self.ENCODINGS_PATH, "rb").read())
        data = np.array(data)
        encodings = [d["encoding"] for d in data]
        names=[d["name"] for d in data]
        return encodings,names

    # Function to calculate EAR Value
    def eye_aspect_ratio(self,eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # Function to mark attendance
    def mark_attendance(self,name,landmarks):
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        leftEAR = self.eye_aspect_ratio(left_eye)
        rightEAR = self.eye_aspect_ratio(right_eye)

		# average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        if ear < self.EYE_AR_THRESH:
            # Check if entry already exists or not
            temp = self.ATTENDANCE_DATA.copy()
            temp = temp[temp['Name'] == name]
            if len(temp)<1:
                # No entry exists, create new entry
                data = {
                    'Name' : name,
                    'Check-In' : datetime.now(),
                    'Check-Out' : None
                }
                self.ATTENDANCE_DATA = self.ATTENDANCE_DATA.append(data,ignore_index = True)
                print("INFO: Marked checkin for {}".format(name))
            else:
                # Entries exist, check if last entry has checked out or not
                if pd.isna(temp.iloc[-1]['Check-Out']):
                    # Checkout from last entry
                    self.ATTENDANCE_DATA.loc[self.ATTENDANCE_DATA[self.ATTENDANCE_DATA['Name'] == name].iloc[-1].name,'Check-Out'] = datetime.now()
                    print("INFO: Marked checkout for {}".format(name))
                else:
                    # Create new Checkin Entry 
                    data = {
                        'Name' : name,
                        'Check-In' : datetime.now(),
                        'Check-Out' : None
                    }
                    self.ATTENDANCE_DATA = self.ATTENDANCE_DATA.append(data,ignore_index = True)
                    print("INFO: Marked checkin for {}".format(name))
        else:
            print("INFO: Eye Blink required to mark attendance.")


    # Function to run live feed for attendance capturing
    def live_feed(self):
        if self.ENCODINGS is None:
            print("ERROR: No Encodings found. Cannot predict any faces.")
            print("-----------------------------------------------------------------------------------------------------------------------------------")
        else:
            video_capture = cv2.VideoCapture(0)
            while True:
                ret, frame = video_capture.read()
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                
                # Find all the faces and face encodings in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                landmarks = face_recognition.face_landmarks(rgb_small_frame,face_locations)

                face_names = []
                for landmark,face_encoding in zip(landmarks,face_encodings):
                    matches = face_recognition.compare_faces(self.ENCODINGS, face_encoding)
                    name = "Unknown" # Default Unknown Face
                    
                    # Find Name with the closest distance
                    face_distances = face_recognition.face_distance(self.ENCODINGS, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.NAMES[best_match_index]

                    face_names.append(name)
                    
                    # Mark Attendance if face found
                    if name != "Unknown":
                        self.mark_attendance(name,landmark)

                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                cv2.imshow('Live Feed', frame)
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Webcam Cleanup
            video_capture.release()
            cv2.destroyAllWindows()

            # Saving Attendance File
            self.ATTENDANCE_DATA.to_excel(self.ATTENDANCE_FILE,index=False)
            print("File saved")

if __name__ == "__main__":
    attObj = Attendance()
    attObj.live_feed()