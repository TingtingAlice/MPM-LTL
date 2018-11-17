"""Refactor file directories, save/rename images and partition the 
train/val/test set, in order to support the unified dataset interface.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')
import time

from zipfile import ZipFile
import os.path as osp
import numpy as np
import scipy.io as sio
import matplotlib.image as mpimg

from bpm.utils.utils import may_make_dir
from bpm.utils.utils import save_pickle
from bpm.utils.utils import load_pickle

from bpm.utils.dataset_utils import get_im_names
from bpm.utils.dataset_utils import partition_train_val_set
from bpm.utils.dataset_utils import new_im_name_tmpl
from bpm.utils.dataset_utils import parse_im_name as parse_new_im_name
from bpm.utils.dataset_utils import move_ims, move_ims_2
from bpm.utils.dataset_utils import change_im_names
from bpm.utils.dataset_utils import anchor_positive_negative_2
from bpm.utils.dataset_utils import mess_up_apn


def parse_original_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = -1 if im_name.startswith('-1') else int(im_name[:4])
  else:
    parsed = int(im_name[4]) if im_name.startswith('-1') \
      else int(im_name[6])
  return parsed


def save_images(original_file, save_dir=None, train_test_split_file=None):
  """Rename and move all used images to a directory."""

  # print("Extracting zip file")
  root = osp.dirname(osp.abspath(original_file))
  if save_dir is None:
    save_dir = root
  may_make_dir(osp.abspath(save_dir))
  # with ZipFile(original_file) as z:
  #   z.extractall(path=save_dir)
  # print("Extracting zip file done")

  new_im_dir = osp.join(save_dir, 'images')
  may_make_dir(osp.abspath(new_im_dir))
  raw_dir = osp.abspath(original_file)
  print('raw_dir: ',raw_dir)
 

  im_paths = []
  nums = []

  im_paths_ = get_im_names(osp.join(raw_dir, 'bounding_box_train'), pattern='*.png',
                           return_path=True, return_np=False)
  im_paths_.sort()
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))
  print('dir_name:   bounding_box_train')
  print('nums:   ',nums)

  # Create (anchor, positive, negative)
  anchor_positive_negative_2(im_paths, parse_original_im_name, save_dir)

  im_paths_ = get_im_names(osp.join(raw_dir, 'bounding_box_test'), pattern='*.png',
                           return_path=True, return_np=False)
  im_paths_.sort()
  im_paths_ = [p for p in im_paths_ if not osp.basename(p).startswith('-1')]
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))
  print('dir_name:   bounding_box_test')
  print('nums:   ',nums)

  im_paths_ = get_im_names(osp.join(raw_dir, 'query'), pattern='*.png',
                           return_path=True, return_np=False)
  im_paths_.sort()
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))
  q_ids_cams = set([(parse_original_im_name(osp.basename(p), 'id'),
                     parse_original_im_name(osp.basename(p), 'cam'))
                    for p in im_paths_])
  print('dir_name:   query')
  print('nums:   ',nums)


  im_paths_ = get_im_names(osp.join(raw_dir, 'gt_bbox'), pattern='*.png',
                           return_path=True, return_np=False)
  im_paths_.sort()
  #print('len of im_paths:'+str(len(im_paths)))
  # Only gather images for those ids and cams used in testing.
  im_paths_ = [p for p in im_paths_
               if (parse_original_im_name(osp.basename(p), 'id'),
                   parse_original_im_name(osp.basename(p), 'cam'))
               in q_ids_cams]
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))
  print('dir_name:   gt_bbox')
  print('nums:   ',nums)

  im_names = move_ims_2(
    im_paths, parse_original_im_name, new_im_name_tmpl)

  split = dict()
  keys = ['trainval_im_names', 'gallery_im_names', 'q_im_names', 'mq_im_names']
  inds = [0] + nums
  print('inds:   ',inds)
  inds = np.cumsum(np.array(inds))
  print('inds:   ',inds)
  print('enumerate(keys):   ',enumerate(keys))
  for i, k in enumerate(keys):
    print('i,k: ',i,k)
    split[k] = im_names[inds[i]:inds[i + 1]]

  save_pickle(split, train_test_split_file)
  print('Saving images done.')

  return split


def transform(original_file, save_dir=None):
  """Refactor file directories, rename images and partition the train/val/test 
  set.
  """

  train_test_split_file = osp.join(save_dir, 'train_split.pkl')
  train_test_split = save_images(original_file, save_dir, train_test_split_file)
  # train_test_split = load_pickle(train_test_split_file)

  # partition train/val/test set

  trainval_ids = list(set([parse_new_im_name(n, 'id')
                           for n in train_test_split['trainval_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  trainval_ids.sort()
  trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
  partitions = partition_train_val_set(
    train_test_split['trainval_im_names'], parse_new_im_name, num_val_ids=100)
  train_im_names = partitions['train_im_names']
  train_ids = list(set([parse_new_im_name(n, 'id')
                        for n in partitions['train_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  train_ids.sort()
  train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

  # change anchor_positive_negative to new names
  apn_pkl_file = osp.join(save_dir, 'anchor_positive_negative_split_2.pkl')
  apn_pkl = load_pickle(apn_pkl_file)
  anchor_list = apn_pkl['anchor']
  positive_list = apn_pkl['positive']
  negative_list = apn_pkl['negative']
  anchor_array = np.array(anchor_list)
  # mess up the order of anchor_positive_negative (trainval)
  new_tv_anchor_names, new_tv_positive_names, new_tv_negative_names = mess_up_apn(anchor_list, positive_list, negative_list)
  # select the train anchor_positive_negative
  train_anchor_names = []
  train_positive_names = []
  train_negative_names = []
  for i in range(train_im_names.shape[0]):
    k = np.where(anchor_array==train_im_names[i])
    ind = k[0][0]
    ta = anchor_list[ind]
    tp = positive_list[ind]
    tn = negative_list[ind]
    train_anchor_names.append(ta)
    train_positive_names.append(tp)
    train_negative_names.append(tn)
  
  # A mark is used to denote whether the image is from
  #   query (mark == 0), or
  #   gallery (mark == 1), or
  #   multi query (mark == 2) set

  val_marks = [0, ] * len(partitions['val_query_im_names']) \
              + [1, ] * len(partitions['val_gallery_im_names'])
  val_im_names = list(partitions['val_query_im_names']) \
                 + list(partitions['val_gallery_im_names'])

  test_im_names = list(train_test_split['q_im_names']) \
                  + list(train_test_split['mq_im_names']) \
                  + list(train_test_split['gallery_im_names'])
  test_marks = [0, ] * len(train_test_split['q_im_names']) \
               + [2, ] * len(train_test_split['mq_im_names']) \
               + [1, ] * len(train_test_split['gallery_im_names'])

  partitions = {'trainval_anchor_im_names': new_tv_anchor_names,
                'trainval_positive_im_names': new_tv_positive_names,
				        'trainval_negative_im_names': new_tv_negative_names,
                'trainval_ids2labels': trainval_ids2labels,
                'train_anchor_im_names': train_anchor_names,
				        'train_positive_im_names': train_positive_names,
				        'train_negative_im_names': train_negative_names,
                'train_ids2labels': train_ids2labels,
                'val_im_names': val_im_names,
                'val_marks': val_marks,
                'test_im_names': test_im_names,
                'test_marks': test_marks}
  partition_file = osp.join(save_dir, 'new_shuffle_apn_partitions_2.pkl')
  save_pickle(partitions, partition_file)
  print('Partition file saved to {}'.format(partition_file))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform Tri-Market Dataset")
  parser.add_argument('--original_file', type=str,
                      default='/GPUFS/nsccgz_ywang_1/dengchufu/data/Market_3_retain/Market_3_extend_trans_end_0')
  parser.add_argument('--save_dir', type=str,
                      default='/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0')
  args = parser.parse_args()
  original_file = osp.abspath(osp.expanduser(args.original_file))
  save_dir = osp.abspath(osp.expanduser(args.save_dir))
  transform(original_file, save_dir)
