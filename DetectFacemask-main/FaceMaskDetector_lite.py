import os
import tensorflow as tf
# from tensorflow import keras
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2 as cv
import cv2
from qr_scan import read_barcodes
import video
import datetime, time
from PIL import Image, ImageOps
import random
import string
import time
from pyzbar import pyzbar
import serial
import _thread

Arduino_Serial = serial.Serial('COM6', 9600)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR)

class FaceMask:

    LABELS = []
    cascade = None
    # model = None
    interpreter = None
    size = (224, 224)
    flag = True

    
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(os.path.join(CURRENT_DIR, "cascade.xml"))
        if(self.cascade.empty()):
            print("cascade empty")

        self.getLabels()

        # modelFile = os.path.join(CURRENT_DIR, "keras_model.h5")
        modelFile = os.path.join(CURRENT_DIR, "model.tflite")
        if(os.path.exists(modelFile)):
            # self.model = tf.keras.models.load_model(modelFile)
            interpreter = Interpreter(modelFile)
            interpreter.allocate_tensors()
        
        #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        np.set_printoptions(suppress=True)

    ####################################################################################################
    
    def getLabels(self):
        with open(os.path.join(CURRENT_DIR, "labels_facemask.txt"), 'r') as file:
            for x in file:
                self.LABELS.append(str(x).replace("\n", ""))
        print(self.LABELS)

    ####################################################################################################

    def TFpredictImgPath(self, imgePath):
        pilImg = image.load_img(imgePath)    

        return self.TFpredictPilImg(pilImg)

    ####################################################################################################

    def TFpredictPilImg(self, pilImg):
        if(self.interpreter == None):
            print("model is null")
            return None
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        image = ImageOps.fit(pilImg, self.size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        # predictions = self.model.predict(data)
        input_tensor = data[0].astype('float32')

        input_index = self.interpreter.get_input_details()[0]['index']
        self.interpreter.set_tensor(input_index, input_tensor)
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()[0]
        output = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        result = np.argmax(output)
        return result

    ####################################################################################################

    def PredictMat(self, mat):
        img = cv.cvtColor(mat, cv.COLOR_BGR2RGB)    
        img = cv.resize(img, self.size)
        img_pil = Image.fromarray(img)

        result = self.TFpredictPilImg(img_pil)
        return result

    ####################################################################################################

    def DetectFaceInFrame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)

        newWidth = int(gray.shape[1] /2)
        newHeight = int(gray.shape[0] /2)

        gray = cv2.resize(gray, (newWidth, newHeight))
        rects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(20,20))

        if(len(rects) > 0):
            rects[:,2:] += rects[:,:2]

        for r in rects:
            r[0] *= 2
            r[1] *= 2
            r[2] *= 2
            r[3] *= 2
            

        return rects

    ####################################################################################################

    def CropMat(self, frame, rect):
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y

        frame = frame[y:h, x:w]
        return frame

    ####################################################################################################

    def GenerateRandomString(self):
        return ''.join(random.choices(string.ascii_lowercase + "_" + string.ascii_uppercase +  string.digits, k=10))

    ####################################################################################################

    def read_barcodes(self, frame):
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            x, y , w, h = barcode.rect
            barcode_info = barcode.data.decode('utf-8')
            cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, barcode_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)

            with open("barcode_result.txt", mode ='w') as file:
                file.write("Recognized Barcode:" + barcode_info)
                # print(barcode_info)
        
        # return barcode_info

    ####################################################################################################

    def skip(self):
        
        time.sleep(10)
        self.flag = True

    ####################################################################################################

    def DetectMask(self, frame):
        startTime = time.time()
        saveImage = False #to debug

        # print(Arduino_Serial.readline())

        read_barcodes(frame)
        rects = self.DetectFaceInFrame(frame)
        arr = []
        if(len(rects) > 0):
            rects[:,2:] += rects[:,:2]
            for rect in rects:
                #print(rect)
                matFace = self.CropMat(frame, rect)                
                predicted = self.PredictMat(matFace)
                result = self.LABELS[predicted]

                color = (0, 0, 255)
                
                
                x1 = rect[0]
                y1 = rect[1]
                x2 = rect[2] - x1
                y2 = rect[3] - y1

                elapsed = time.time() - startTime

                if(saveImage):
                    cv2.imwrite(result +"\\" + datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S") + "_" + GenerateRandomString() + ".jpg", matFace)

                arr.append(result)

                if(result == "Mask"):
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                elif(result == "Hand" or result == "No mask" or result == "Wrong"):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    if self.flag:
                        Arduino_Serial.write('4'.encode())
                        print("checked")
                        self.flag = False
                        try:
                            _thread.start_new_thread(self.skip, ( ))
                        except:
                            print('error')

                if(result != "Nothing"):
                    cv.putText(frame, str(result), (10,60), cv.FONT_HERSHEY_PLAIN, 3, color, thickness = 2)
                    cv.putText(frame, "{:10.2f} s".format(elapsed), (300,60), cv.FONT_HERSHEY_PLAIN, 3, color, thickness = 2)
        else:
            if(saveImage):
                cv2.imwrite("noface\\" + datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S") + "_" + GenerateRandomString() + ".jpg", frame)

        return frame, arr

####################################################################################################


faceMask = FaceMask()

if __name__ == '__main__':
    cap = video.create_capture(0)
    # cap = cv2.VideoCapture("http://192.168.8.180:81/stream")
    while True:
        
        _ret, frame = cap.read()
        frame, arr = faceMask.DetectMask(frame)

        cv.imshow('frame', frame)
        ch = cv.waitKey(20)
        if ch == 27:
            break