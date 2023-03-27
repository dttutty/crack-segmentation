import os
import os.path as path
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import numpy
import cv2


if __name__ == '__main__':
    ann_dir = '/home/zhaolei/Projects/mmsegmentation/data/crack/annotations/validation'
    img_dir = '/home/zhaolei/Projects/mmsegmentation/data/crack/images/validation'
    vis_dir = '/home/zhaolei/Projects/mmsegmentation/data/crack/vis/validation'
    
    listdir = os.listdir(ann_dir)
    
    for filename in listdir:
        img = cv2.imread(path.join(img_dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ann = cv2.imread(path.join(ann_dir, filename))
        ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        
        red = numpy.array([120,0,0],dtype='uint8')
        vis = cv2.add(img, ann*red)
        
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path.join(vis_dir, filename), vis)