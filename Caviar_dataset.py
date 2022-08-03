import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import sys
from xml.etree import ElementTree
import pylab as pl


class CaviarDataset():

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)


    def prepare(self, class_map=None):

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def load_dataset(self, data_dir, dataset_dir, obj_class, is_train=True):
        self.add_class(dataset_dir, 1, obj_class)
        annotations_dir = data_dir + fr'\annots'
        for im_id, filename in enumerate(os.listdir(dataset_dir)):
            img_path = os.path.join(dataset_dir, filename)
            print(img_path)
            annots = os.path.join(annotations_dir, os.listdir(annotations_dir)[0])
            self.add_image('dataset', image_id=im_id, path=img_path, annotation=annots)

    def extract_boxes(self, filename, im_id,all_roles,all_context):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes, all_labels = [], []
        for f, frames in enumerate(root.findall('frame')):
            xc, yc, h, w, sits, roles, movements, contexts = [], [], [], [], [], [], [], []
            if f == im_id:
                for box in frames.findall('.//box'):
                    xc.append(box.get('xc'))
                    yc.append(box.get('yc'))
                    h.append(box.get('h'))
                    w.append(box.get('w'))
                for role in frames.findall('.//role'):
                    roles.append(role.text)
                for context in frames.findall('.//context'):
                    contexts.append(context.text)

                coors = np.ones(shape=(len(xc), 4))
                a, b = coors.shape
                for i in range(a):
                    coors[i, :] = [h[i], w[i], xc[i], yc[i]]
                    all_labels.append([all_roles[roles[i]], all_context[contexts[i]]])
                boxes.append(coors)
        return boxes, all_labels
