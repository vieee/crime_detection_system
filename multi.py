import os
import copy
import time
import pickle
import cv2
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping
from multiprocessing import Process, Queue
import smtplib

s = smtplib.SMTP('smtp.gmail.com', 587)
model = load_model('model_resnet_gru_10fps_browse0_3.h5')
# s.starttls()
# s.login("modgtrek197@gmail.com", "!4BMW%ZV$C!Pz!!4GcsP4fwK")







#Change the paths in the program


def face_detector(pqueue):
	faceCascade = cv2.CascadeClassifier('E:\\Programs\\anaconda3\\envs\\MP\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
	data = pickle.loads(open('E:\\College\\Projects\\FourthYear\\SEM7\\SuspiciousBehaviourRecognition\\FaceDetector-Harshita\\face_enc', "rb").read())
	face_flag = True
	print("Streaming started")
	process_this_frame = True

	while True:
	    frame = pqueue.get()
	    if(frame =="done"):
	    	break
	    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
	    rgb_small_frame = small_frame[:, :, ::-1]
	    if process_this_frame:
	        face_locations = face_recognition.face_locations(rgb_small_frame)
	        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
	        face_names = []
	        for face_encoding in face_encodings:
	            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
	            name = "Unknown"

	            if True in matches:
	                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
	                counts = {}
	                for i in matchedIdxs:
	                    name = data["names"][i]
	                    counts[name] = counts.get(name, 0) + 1

	                name = max(counts, key=counts.get)
	            face_names.append(name)
	            if face_names and face_flag:
	            	name_list = ""
	            	for names in face_names:
	            		name_list += " "+ names
	            	print("sadf")
	            	message = "Person Detected " + name_list
	            	# s.sendmail("modgtrek197@gmail.com", "rajatshenoy@gmail.com", message)
	            	face_flag = False


	    process_this_frame = not process_this_frame

	    for (top, right, bottom, left), name in zip(face_locations, face_names):
	        top *= 5
	        right *= 5
	        bottom *= 5
	        left *= 5
	        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	        font = cv2.FONT_HERSHEY_DUPLEX
	        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	    cv2.namedWindow('Face Detector')
	    cv2.moveWindow('Face Detector', 1300,30) 
	    cv2.imshow('Face Detector', frame)

	    # Hit 'q' on the keyboard to quit!
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

def object_detector(pqueue):

	object_flag = True
	thres = 0.45  # Threshold to detect object
	nms_threshold = 0.4

	classNames = []
	classFile = "E:\\College\\Projects\\FourthYear\\SEM7\\SuspiciousBehaviourRecognition\\ObjectDetector-Deepak\\coco.names"
	with open(classFile, "rt") as f:
	    classNames = f.read().rstrip("\n").split("\n")

	configPath = "E:\\College\\Projects\\FourthYear\\SEM7\\SuspiciousBehaviourRecognition\\ObjectDetector-Deepak\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
	weightsPath = "E:\\College\\Projects\\FourthYear\\SEM7\\SuspiciousBehaviourRecognition\\ObjectDetector-Deepak\\frozen_inference_graph.pb"

	net = cv2.dnn_DetectionModel(weightsPath, configPath)
	net.setInputSize(320, 320)
	net.setInputScale(1.0 / 127.5)
	net.setInputMean((127.5, 127.5, 127.5))
	net.setInputSwapRB(True)
	video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

	while True:
		ret, img = video_capture.read()
		img1 = copy.copy(img)
		pqueue.put(img1)
		
		classIds, confs, bbox = net.detect(img, confThreshold=thres)
		bbox = list(bbox)
		confs = list(np.array(confs).reshape(1, -1)[0])
		confs = list(map(float, confs))
		indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

		
		for i in indices:
			i = i[0]
			box = bbox[i]
			x, y, w, h = box[0], box[1], box[2], box[3]
			cv2.rectangle(img, (x, y), (x+w, h+y), color=(0, 255, 0), thickness=2)
			cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
			if classNames[classIds[i][0]-1] == "knife":
				if object_flag:	
		 			message = "Knife Detected"
		 			print("ss")
		 			# s.sendmail("modgtrek197@gmail.com", "rajatshenoy@gmail.com", message)
		 			object_flag = False		
		
		cv2.namedWindow('Object Detector')
		cv2.moveWindow('Object Detector', 600,30) 
		cv2.imshow("Object Detector", img)
		k = cv2.waitKey(1) & 0xff
		if k==27:
			break

	pqueue.put('done')
	video_capture.release()
	cv2.destroyAllWindows()

def display_all_suspicious_images(video, labels):
    sus_flag = True
    img = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(video.shape[0]):
        for j in range(video.shape[1]):
            image = copy.copy(video[i,j,:,:,:])
            label = labels[i,j,:]
            im_shape = image.shape
            if label >= 0.5:
                image = cv2.rectangle(image, (1, 1), (im_shape[1]-1, im_shape[0]-1) , [0, 0, 255], 3)
                image = cv2.putText(img=image,text='Suspicious Behavior', org=(24,24), fontFace=font ,fontScale=0.4, color=[0,0,255], lineType=2)
                if sus_flag:
                	message = "Suspicious Behaviour Detected "
                	print("Sssss")
                	# s.sendmail("modgtrek197@gmail.com", "rajatshenoy@gmail.com", message)
                	sus_flag = False	
            else:
                image = cv2.putText(img=image,text='Not Suspicious Behavior', org=(24,24), fontFace=font ,fontScale=0.4, color=[255,0,0], lineType=3)

            image = cv2.resize(image, (500, 500))

            cv2.namedWindow('Suspicious Behavior')
            cv2.moveWindow('Suspicious Behavior', 40,30) 
            cv2.imshow('Suspicious Behavior',image)
            if id == 1000:
                break
                pl.close('all')

            if cv2.waitKey(0) == ord('q'):
                continue
            else:
            	break

def test_model(file_name):
	
	earlystop = EarlyStopping(patience=7)
	callbacks = [earlystop]
	with open(file_name, 'rb') as f:
		test = np.load(f, allow_pickle=True)
	test = test[:,10:285,30:335]
	test = np.array([cv2.resize(i, (224,224)) for i in test[:,:]])
	j = len(test) % 10
	k = len(test)- j
	test = test[j:]
	test = test.reshape(k//10,10 ,224, 224, 3)
	labels = model.predict(test,batch_size=10,callbacks=callbacks)
	display_all_suspicious_images(test,labels)

if __name__ == "__main__":

	print("ID of main process: {}".format(os.getpid()))
	pqueue = Queue()
	# p1 = Process(target=test_model,args=("Caviar-Dataset/Test/Leftbox.npy",))
	# p1 = Process(target=test_model,args=("Caviar-Dataset/Test/Browse.npy",))
	p1 = Process(target=test_model,args=("Caviar-Dataset/Test/Fight.npy",))

	p2 = Process(target=face_detector,args=(pqueue,))
	p3 = Process(target=object_detector,args=(pqueue,))

	p1.start()
	p2.start()
	p3.start()
  
	print("ID of process p1: {}".format(p1.pid))
	print("ID of process p2: {}".format(p2.pid))
	print("ID of process p3: {}".format(p3.pid))

	p1.join()
	p2.join()
	p3.join()
	s.quit()
	print("processes finished execution!")