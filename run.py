import cv2
import face_recognition
import pickle
import os
import numpy as np
import pandas as pd

class Attendance(object):
    def __init__(self):
        self.ENCODINGS_PATH = "encodings.pkl"

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

    # Function to mark attendance
    def mark_attendance(self):
        pass

    # Function to run live feed for attendance capturing
    def live_feed(self):
        if self.ENCODINGS is None:
            print("ERROR: No Encodings found. Cannot predict any faces.")
            print("-----------------------------------------------------------------------------------------------------------------------------------")
        else:
            video_capture = cv2.VideoCapture(0)
            process_this_frame = True

            while True:
                ret, frame = video_capture.read()
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
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
                            print("INFO: Marking Attendance for {}".format(name))
                            self.mark_attendance(name)


                process_this_frame = not process_this_frame

                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                cv2.imshow('Live Feed', frame)
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Webcam Cleanup
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    attObj = Attendance()
    attObj.live_feed()