# --------------------------------------------------------
# Faster RCNN or RFCN for objection detection
# Written by Yong Du
# --------------------------------------------------------

import os
import datasets
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class our_dataset(imdb):
    def __init__(self, image_set, data_path, img_list, label_list):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._data_path = data_path
        self._img_list = img_list
        self._label_list = label_list
        self._classes = ('__background__', # always index 0
                         'car', 'person', 'bicycle', 'moto_bicycle', 'tricycle', 'moto_tricycle')
        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))

        assert os.path.exists(self._data_path), \
                'dataset path does not exist: {}'.format(self._data_path)
        self._image_index = self._load_image_label_list(self._img_list)
        self._label_index = self._load_image_label_list(self._label_list)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_label_list(self, file_list):
        """
        Load the indexes listed in this dataset's image set file.
        """
        list_file_path = os.path.join(self._data_path, file_list)
        assert os.path.exists(list_file_path), \
                'Path does not exist: {}'.format(list_file_path)
        with open(list_file_path) as f:
            file_name_list = [x.strip() for x in f.readlines()]
        return file_name_list

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_dataset_annotation(index)
                    for index in self._label_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_dataset_annotation(self, index):
        """
        Load image and bounding boxes info.
        """
        filename = os.path.join(self._data_path, index)
        boxes = []
        gt_classes = []
        overlaps = []
        with open(filename) as f:
            lines = [l.strip() for l in f.readlines()]
            for l in lines:
                box_info = l.split()
                if str(box_info[0]).lower() in self._classes:
                    boxes.append([float(box_info[1]), float(box_info[2]),
                                  float(box_info[3]), float(box_info[4])])
                    cls = self._class_to_ind[str(box_info[0]).lower()]
                    gt_classes.append(cls)
                    overlap = np.zeros(self.num_classes, dtype=np.float32)
                    overlap[cls] = 1.0
                    overlaps.append(overlap)
 
        overlaps = scipy.sparse.csr_matrix(np.array(overlaps))

        return {'boxes': np.array(boxes),
                'gt_classes': np.array(gt_classes),
                'gt_overlaps': overlaps,
                'flipped' : False} 

if __name__ == '__main__':
    d = datasets.our_dataset('our_dataset', '')
    res = d.roidb
    from IPython import embed; embed()
