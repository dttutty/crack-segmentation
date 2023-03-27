# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import gzip
import os
import os.path as osp
import tarfile
import tempfile
import mmcv
import cv2
import numpy
STARE_LEN = 20
TRAINING_LEN = 10


def un_gz(src, dst):
    g_file = gzip.GzipFile(src)
    with open(dst, 'wb+') as f:
        f.write(g_file.read())
    g_file.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert STARE dataset to mmsegmentation format')
    parser.add_argument('-i', '--input_dir', default='/home/zhaolei/datasets/SEGMENTATION/crack_segmentation_dataset', help='the path of stare-images.tar')
    parser.add_argument('-o', '--out_dir', default='/home/zhaolei/Projects/mmsegmentation/data/crack', help='output path')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    input_dir = args.input_dir
    if args.out_dir is None:
        out_dir = osp.join('data', 'crack')
    else:
        out_dir = args.out_dir
        
    img_mask_basename_pairs = [{'src':'images','dst':'images'},{'src':'masks','dst':'annotations'}]
    train_test_basename_pairs = [{'src':'train','dst':'training'},{'src':'test','dst':'validation'}]
    
    print('Making directories...')

    mmcv.mkdir_or_exist(out_dir)
    for i in img_mask_basename_pairs:
        mmcv.mkdir_or_exist(osp.join(out_dir, i['dst']))
        for j in train_test_basename_pairs:
            src, dst = osp.join(input_dir, j['src'], i['src']), osp.join(out_dir, i['dst'], j['dst'])
            if i['src'] == 'masks':
                mmcv.mkdir_or_exist(dst)
                for img_name in os.listdir(src):
                    img = cv2.imread(osp.join(src, img_name))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = numpy.expand_dims(img, axis=2)
                    cv2.imwrite(osp.join(dst, img_name), img[:,:,0]//128)
            else:
                os.symlink(src, dst)
                pass

    print('Done!')


if __name__ == '__main__':
    main()