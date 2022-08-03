
import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import sys
from xml.etree import ElementTree
import pylab as pl
from Caviar_dataset import CaviarDataset


def extract_frames_of_video(video_path, new_frame_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    if os.path.exists(new_frame_path):
        shutil.rmtree(new_frame_path)
        os.mkdir(new_frame_path)
    else:
        os.mkdir(new_frame_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(new_frame_path + r'\frame{:d}.jpg'.format(count), frame)
            count += 1
            cap.set(25, count)
        else:
            cap.release()
            cv2.destroyAllWindows()
            break


def run_from_video(data_dir, frame_path, action_name, caviar_obj,all_roles,all_context):
    caviar_obj.load_dataset(data_dir, frame_path, action_name, is_train=True)
    caviar_obj.prepare()
    num_frame = len(os.listdir(frame_path))
    X = []
    y = np.zeros([int((num_frame - 251)/3)+1, 2], dtype=object)

    for i, ids in enumerate(np.arange(250, num_frame - 1, 3)):
        f = os.path.join(frame_path, 'frame{0}.jpg'.format(ids))
        image = pl.imread(f)
        X.append(image)

        path = caviar_obj.image_info[ids]['annotation']
        box, mov = caviar_obj.extract_boxes(path, ids,all_roles,all_context)

        if len(box) == 0:
            break
        box = np.array(box[0])
        mov = np.array(mov)
        temp = np.zeros([1, len(box)], dtype=object)

        if len(mov) > 0:
            for w in range(len(box)):
                temp[0, w] = mov[w]
            y[i, 0] = [z[0] for z in temp[0]]
            y[i, 1] = [z[1] for z in temp[0]]  
        else:
            y[i,:] = [],[]
    return X,y


def suspicious_behavior_labels(labels):
    very_suspicious_index = 2
    suspicious_index = 1
    not_suspicious_index = 0
    num_frames = len(labels)

    new_labels = np.array([],dtype=int)
    check_role = np.array([])
    check_context = np.array([])

    for i in range(num_frames):
        if labels[i,0] != [] and labels[i,1] != []:
            check_role = np.append(check_role, np.max(labels[i,0]))
            check_context = np.append(check_context, np.max(labels[i,1]))
        else:
            check_role = np.append(check_role, 0)
            check_context = np.append(check_context, 0)

    for i in range(num_frames):
        if check_role[i] == very_suspicious_index  or check_context[i] == very_suspicious_index:
            new_labels = np.append(new_labels, 2)
        elif check_role[i] == suspicious_index  or check_context[i] == suspicious_index:
            new_labels = np.append(new_labels, 1)
        else:
            new_labels = np.append(new_labels, 0)

    return new_labels.reshape(len(new_labels), 1)


if __name__ == '__main__':

    # Change this string to the path of the dataset
    main_dir = r'E:\College\Projects\FourthYear\SEM7\SuspiciousBehaviourRecognition\Data'

    actions = ['Browse', 'Fight', 'Groups_Meeting', 'LeftBag', 'Rest', 'Walk']
    all_roles = {'fighters': 2, 'fighter': 2, 'leaving object': 2, 'browser': 1, 'browsers': 1, 'walkers': 0, 'meet': 0, 'meeters': 0, 'walker': 0}
    all_context = {'fighting': 2, 'leaving': 2, 'drop down': 2, 'browsing': 1, 'immobile': 0, 'walking': 0, 'meeting': 0, 'windowshop': 0, 'shop enter': 0, 'shop exit': 0, 'shop reenter': 0, 'none':0}

    
    for id_dataset in range(6):
        # id_dataset = int(input('Choose action'))
        action_dir = os.path.join(main_dir, actions[id_dataset])

        train_testX, train_testy = [], []
        for id_vid in range(1,4):
            data_dir = os.path.join(action_dir, actions[id_dataset]) + str(id_vid)

            train_set = CaviarDataset()

            new_frames_file = action_dir + '\\' + actions[id_dataset] + str(id_vid) +'\\new' 

            video_file = os.path.join(data_dir + r'\video', os.listdir(data_dir + r'\video')[0])
           
            extract_frames_of_video(os.path.join(data_dir, video_file) , new_frames_file)

            X_frames,y_frames = run_from_video(data_dir, new_frames_file, actions[id_dataset], train_set, all_roles,all_context)

            # print('video number {0} \n X_frames = {1} \n y_frames = {2}'.format(id_vid, len(X_frames), len(y_frames)))

            train_testX.append(np.array(X_frames))
            train_testy.append(np.array(y_frames))


        X = np.concatenate((train_testX[0], train_testX[1], train_testX[2]), axis=0)
        y = np.concatenate((train_testy[0], train_testy[1], train_testy[2]), axis=0)
        y = suspicious_behavior_labels(y)

        with open(main_dir+r'\{0}\X.npy'.format(actions[id_dataset]), 'wb') as f:
            np.save(f,X)
        with open(main_dir+r'\{0}\y.npy'.format(actions[id_dataset]), 'wb') as f:
            np.save(f, y)






