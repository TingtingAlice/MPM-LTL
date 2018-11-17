from __future__ import print_function
import os.path as osp
import time
import numpy as np
import scipy.io as sio
import matplotlib.image as mpimg
import glob
from collections import defaultdict
import shutil
import torch

from utils import may_make_dir
from utils import save_pickle
from utils import load_pickle

new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'

def parse_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed


def get_im_names(im_dir, pattern='*.png', return_np=True, return_path=False):
  """Get the image names in a dir. Optional to return numpy array, paths."""
  im_paths = glob.glob(osp.join(im_dir, pattern))
  im_names = [osp.basename(path) for path in im_paths]
  ret = im_paths if return_path else im_names
  if return_np:
    ret = np.array(ret)
  return ret


def intersection_mask(mask_1, mask_2):
  """the intersection set between mask_1 and mask_2"""
  mask_sum = mask_1 + mask_2
  tem = (mask_sum==2)
  count = mask_sum[tem].size
  return count


def nearest_mask(anchor_mask, sample_, top=1):
  """
  the intersection set between an anchor_mask and the images in sample_
  Args:
       anchor_mask = mpimg.imread(anchor)[:,:,3]
       sample_ = a list of images
  Return:
       sample: the nearest one (to anchor) in sample_
  """
  count_list=[]
  for i in range(len(sample_)):
    sa = sample_[i]
    sa_mask = mpimg.imread(sa)[:,:,3]
    count = intersection_mask(anchor_mask, sa_mask)
    count_list.append(count)
  count_list = np.array(count_list)
  index = np.argsort(count_list) # sort from small to large
  index = index[::-1] # sort from large to small
  sample = []
  for j in range(top):
    sample.append(sample_[index[j]]) # the index of the largest one/Top x (x depend on top)
  return sample


def anchor_positive_negative(im_paths, parse_im_name, save_dir):
  """
  1. compute the intersection set of masks
  2. compute the distance of features
  3. calculate the difference between d_n and d_p
  """
  feat_path='/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/tri/market_30_retain_feat.mat'
  feat_mat=sio.loadmat(feat_path)
  im_names=feat_mat['im_names']
  # h_list=feat_mat['h_list']
  feats=feat_mat['feat']
  im_names2feats=dict(zip(im_names,feats)) # use the new im_names as keys
  name2new = names2newnames(im_paths,  parse_im_name, new_im_name_tmpl) # change the ori_name to new_name

  anchor_list = []
  positive_list = []
  negative_list = []

  since = time.time()
  localtime = time.asctime( time.localtime(time.time()) )
  print('--- Current Time:    ',localtime)
  print('--- Dividing ...')
  for i in range(len(im_paths)):
    anchor_path = im_paths[i]
    anchor = osp.basename(anchor_path)
    new_an = name2new[anchor] # change anchor to new name
    feat_an = im_names2feats[new_an] # the feature of anchor
    aid = parse_im_name(anchor,'id')
    acam = parse_im_name(anchor,'cam')
    an_mask = mpimg.imread(anchor_path)[:,:,3]
    # the same id but in the different cam:
    positive_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') == aid) & (parse_im_name(osp.basename(p), 'cam') != acam) ] 
    # the different id but in the same cam:
    negative_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') != aid) & (parse_im_name(osp.basename(p), 'cam') == acam) ] 
    # Positive
    positive_path = nearest_mask(an_mask, positive_)
    positive_path = positive_path[0]
    positive = osp.basename(positive_path)
    new_po = name2new[positive] # change positive to new name
    feat_po = im_names2feats[new_po] # the feature of positive
    # Negative
    negative_path = nearest_mask(an_mask, negative_, top=20)
    nega_tem_list = []
    feat_nega_list = []
    for j in range(20):
      negative = osp.basename(negative_path[j])
      new_ne = name2new[negative] # change negative to new name
      nega_tem_list.append(new_ne)
      feat_ne = im_names2feats[new_ne] # the feature of positive
      feat_nega_list.append(feat_ne)
    hard_index = hard_sample(feat_an, feat_po, feat_nega_list, small=5)
    for k in range(5):
      anchor_list.append(new_an)
      positive_list.append(new_po)
      negative_list.append(nega_tem_list[hard_index[k]])
  time_elapsed = time.time() - since
  print('--- Dividing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  apn=dict(anchor=anchor_list,positive=positive_list,negative=negative_list)
  anchor_positive_negative_file = osp.join(save_dir, 'anchor_positive_negative_split.pkl')
  save_pickle(apn, anchor_positive_negative_file)


def anchor_positive_negative_2(im_paths, parse_im_name, save_dir):
  """
  compute the distance of features first, and then calculate the difference between d_n and d_p, last compute the intersection set of masks
  """
  feat_path='/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/tri/market_30_retain_feat.mat'
  feat_mat=sio.loadmat(feat_path)
  im_names=feat_mat['im_names']
  # h_list=feat_mat['h_list']
  feats=feat_mat['feat']
  im_names2feats=dict(zip(im_names,feats)) # use the new im_names as keys
  old2new = names2newnames(im_paths,  parse_im_name, new_im_name_tmpl) # change the ori_name to new_name

  anchor_list = []
  positive_list = []
  negative_list = []

  since = time.time()
  localtime = time.asctime( time.localtime(time.time()) )
  print('--- Current Time:    ',localtime)
  print('--- Dividing ...')
  for i in range(len(im_paths)):
    anchor_path = im_paths[i]
    anchor = osp.basename(anchor_path)
    new_an = old2new[anchor] # change anchor to new name
    feat_an = im_names2feats[new_an] # the feature of anchor
    aid = parse_im_name(anchor,'id')
    acam = parse_im_name(anchor,'cam')
    an_mask = mpimg.imread(anchor_path)[:,:,3]
    # the same id but in the different cam:
    positive_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') == aid) & (parse_im_name(osp.basename(p), 'cam') != acam) ] 
    # the different id but in the same cam:
    negative_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') != aid) & (parse_im_name(osp.basename(p), 'cam') == acam) ] 
    ### Positive
    # posi_tem_list = []
    feat_posi_list = []
    for p in range(len(positive_)):
      positive = osp.basename(positive_[p])
      new_po_t = old2new[positive] # change positive to new name
      # posi_tem_list.append(new_po_t)
      feat_po_t = im_names2feats[new_po_t] # the feature of positive
      feat_posi_list.append(feat_po_t)
    dist_ap = dist2(feat_an, feat_posi_list) # compute the distance of feature between anchor and positive_
    index = np.argsort(dist_ap)
    index_ap = index[0:5]
    posi_list = []
    for q in range(len(index_ap)):
      posi_list.append(positive_[index_ap[q]]) # select five postive candidates which distances of feature are the farthest five
    positive_path = nearest_mask(an_mask, posi_list) # choose the positive from five positive candidates
    positive_path = positive_path[0]
    positive = osp.basename(positive_path)
    new_po = old2new[positive] # change positive to new name
    feat_po = im_names2feats[new_po] # the feature of positive
    ### Negative
    # nega_tem_list = []
    feat_nega_list = []
    for j in range(len(negative_)):
      negative = osp.basename(negative_[j])
      new_ne_t = old2new[negative] # change negative to new name
      # nega_tem_list.append(new_ne_t)
      feat_ne_t = im_names2feats[new_ne_t] # the feature of negative
      feat_nega_list.append(feat_ne_t)
    ### Hard Sample
    hard_index = hard_sample(feat_an, feat_po, feat_nega_list, small=20) # compute the distance of feature between anchor and negative_
    # print(len(hard_index))
    negative_tem_path = []
    for k in range(20):
      negative_tem_path.append(negative_[hard_index[k]])
    negative_path = nearest_mask(an_mask, negative_tem_path, top=5)
    for k in range(5):
      anchor_list.append(new_an)
      positive_list.append(new_po)
      negative = osp.basename(negative_path[k])
      new_ne = old2new[negative] # change positive to new name
      negative_list.append(new_ne)
  time_elapsed = time.time() - since
  print('--- Dividing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  apn=dict(anchor=anchor_list,positive=positive_list,negative=negative_list)
  anchor_positive_negative_file = osp.join(save_dir, 'anchor_positive_negative_split_2.pkl')
  save_pickle(apn, anchor_positive_negative_file)


def anchor_positive_negative_2_2(im_paths, parse_im_name, save_dir):
  """
  compute the distance of features first, and then calculate the difference between d_n and d_p, last compute the intersection set of masks
  """
  feat_path='/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/tri/market_30_retain_feat.mat'
  feat_mat=sio.loadmat(feat_path)
  im_names=feat_mat['im_names']
  # h_list=feat_mat['h_list']
  feats=feat_mat['feat']
  im_names2feats=dict(zip(im_names,feats)) # use the new im_names as keys
  old2new = names2newnames(im_paths,  parse_im_name, new_im_name_tmpl) # change the ori_name to new_name

  anchor_list = []
  positive_list = []
  negative_list = []

  since = time.time()
  localtime = time.asctime( time.localtime(time.time()) )
  print('--- Current Time:    ',localtime)
  print('--- Dividing ...')
  for i in range(len(im_paths)):
    anchor_path = im_paths[i]
    anchor = osp.basename(anchor_path)
    new_an = old2new[anchor] # change anchor to new name
    feat_an = im_names2feats[new_an] # the feature of anchor
    aid = parse_im_name(anchor,'id')
    acam = parse_im_name(anchor,'cam')
    an_mask = mpimg.imread(anchor_path)[:,:,3]
    # the same id but in the different cam:
    positive_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') == aid) & (parse_im_name(osp.basename(p), 'cam') != acam) ] 
    # the different id but in the same cam:
    negative_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') != aid) & (parse_im_name(osp.basename(p), 'cam') == acam) ] 
    ### Positive
    # posi_tem_list = []
    feat_posi_list = []
    for p in range(len(positive_)):
      positive = osp.basename(positive_[p])
      new_po_t = old2new[positive] # change positive to new name
      # posi_tem_list.append(new_po_t)
      feat_po_t = im_names2feats[new_po_t] # the feature of positive
      feat_posi_list.append(feat_po_t)
    dist_ap = dist2(feat_an, feat_posi_list) # compute the distance of feature between anchor and positive_
    index = np.argsort(dist_ap)
    index = index[::-1]
    index_ap = index[0:5]
    posi_list = []
    for q in range(len(index_ap)):
      posi_list.append(positive_[index_ap[q]]) # select five postive candidates which distances of feature are the farthest five
    positive_path = nearest_mask(an_mask, posi_list) # choose the positive from five positive candidates
    positive_path = positive_path[0]
    positive = osp.basename(positive_path)
    new_po = old2new[positive] # change positive to new name
    feat_po = im_names2feats[new_po] # the feature of positive
    ### Negative
    # nega_tem_list = []
    feat_nega_list = []
    for j in range(len(negative_)):
      negative = osp.basename(negative_[j])
      new_ne_t = old2new[negative] # change negative to new name
      # nega_tem_list.append(new_ne_t)
      feat_ne_t = im_names2feats[new_ne_t] # the feature of negative
      feat_nega_list.append(feat_ne_t)
    ### Hard Sample
    hard_index = hard_sample(feat_an, feat_po, feat_nega_list, small=20) # compute the distance of feature between anchor and negative_
    # print(len(hard_index))
    negative_tem_path = []
    for k in range(20):
      negative_tem_path.append(negative_[hard_index[k]])
    negative_path = nearest_mask(an_mask, negative_tem_path, top=5)
    for k in range(5):
      anchor_list.append(new_an)
      positive_list.append(new_po)
      negative = osp.basename(negative_path[k])
      new_ne = old2new[negative] # change positive to new name
      negative_list.append(new_ne)
  time_elapsed = time.time() - since
  print('--- Dividing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  apn=dict(anchor=anchor_list,positive=positive_list,negative=negative_list)
  anchor_positive_negative_file = osp.join(save_dir, 'anchor_positive_negative_split_2_2.pkl')
  save_pickle(apn, anchor_positive_negative_file)


def anchor_positive_negative_3(im_paths, parse_im_name, save_dir):
  """
  compute the distance of features first, and then compute the intersection set of masks
  """
  feat_path='/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/tri/market_30_retain_feat.mat'
  feat_mat=sio.loadmat(feat_path)
  im_names=feat_mat['im_names']
  # h_list=feat_mat['h_list']
  feats=feat_mat['feat']
  im_names2feats=dict(zip(im_names,feats)) # use the new im_names as keys
  old2new = names2newnames(im_paths,  parse_im_name, new_im_name_tmpl) # change the ori_name to new_name

  anchor_list = []
  positive_list = []
  negative_list = []

  since = time.time()
  localtime = time.asctime( time.localtime(time.time()) )
  print('--- Current Time:    ',localtime)
  print('--- Dividing ...')
  for i in range(len(im_paths)):
    anchor_path = im_paths[i]
    anchor = osp.basename(anchor_path)
    new_an = old2new[anchor] # change anchor to new name
    feat_an = im_names2feats[new_an] # the feature of anchor
    aid = parse_im_name(anchor,'id')
    acam = parse_im_name(anchor,'cam')
    an_mask = mpimg.imread(anchor_path)[:,:,3]
    # the same id but in the different cam:
    positive_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') == aid) & (parse_im_name(osp.basename(p), 'cam') != acam) ] 
    # the different id but in the same cam:
    negative_ = [p for p in im_paths
                 if (parse_im_name(osp.basename(p), 'id') != aid) & (parse_im_name(osp.basename(p), 'cam') == acam) ] 
    ### Positive
    # posi_tem_list = []
    feat_posi_list = []
    for p in range(len(positive_)):
      positive = osp.basename(positive_[p])
      new_po_t = old2new[positive] # change positive to new name
      # posi_tem_list.append(new_po_t)
      feat_po_t = im_names2feats[new_po_t] # the feature of positive
      feat_posi_list.append(feat_po_t)
    dist_ap = dist2(feat_an, feat_posi_list) # compute the distance of feature between anchor and positive_
    index = np.argsort(dist_ap)
    index = index[::-1]
    index_ap = index[0:5]
    posi_list = []
    for q in range(len(index_ap)):
      posi_list.append(positive_[index_ap[q]]) # select five postive candidates which distances of feature are the farthest five
    positive_path = nearest_mask(an_mask, posi_list) # choose the positive from five positive candidates
    positive_path = positive_path[0]
    positive = osp.basename(positive_path)
    new_po = old2new[positive] # change positive to new name
    feat_po = im_names2feats[new_po] # the feature of positive
    ### Negative
    # nega_tem_list = []
    feat_nega_list = []
    for j in range(len(negative_)):
      negative = osp.basename(negative_[j])
      new_ne_t = old2new[negative] # change negative to new name
      # nega_tem_list.append(new_ne_t)
      feat_ne_t = im_names2feats[new_ne_t] # the feature of negative
      feat_nega_list.append(feat_ne_t)
    ## Hard negative
    dist_an = dist2(feat_an, feat_nega_list) # compute the distance of feature between anchor and negative_
    index = np.argsort(dist_an)
    index_an = index[0:20]
    nega_list = []
    for q in range(len(index_an)):
      nega_list.append(negative_[index_an[q]]) # select five negative candidates which distances of feature are the closest five
    for k in range(2): # 
      anchor_list.append(new_an)
      positive_list.append(new_po)
      negative = osp.basename(nega_list[k])
      nega_list.remove(nega_list[k])
      new_ne = old2new[negative] # change positive to new name
      negative_list.append(new_ne)
    negative_path = nearest_mask(an_mask, nega_list, top=3) # choose 3 negative from 20 negative candidates
    negative_path = negative_path[0:3]
    for k in range(3):
      anchor_list.append(new_an)
      positive_list.append(new_po)
      negative = osp.basename(negative_path[k])
      new_ne = old2new[negative] # change positive to new name
      negative_list.append(new_ne)
  time_elapsed = time.time() - since
  print('--- Dividing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  apn=dict(anchor=anchor_list,positive=positive_list,negative=negative_list)
  anchor_positive_negative_file = osp.join(save_dir, 'anchor_positive_negative_split_3.pkl')
  save_pickle(apn, anchor_positive_negative_file)


def move_ims(ori_im_paths, new_im_dir, parse_im_name, new_im_name_tmpl):
  """Rename and move images to new directory."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_path in ori_im_paths:
    im_name = osp.basename(im_path)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    new_im_path = osp.join(new_im_dir, new_im_name)
    if not osp.isfile(new_im_path):
      shutil.copy(im_path, new_im_path)
    new_im_names.append(new_im_name)
  return new_im_names


def move_ims_2(ori_im_paths, parse_im_name, new_im_name_tmpl):
  """Rename but not move images to new directory."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_path in ori_im_paths:
    im_name = osp.basename(im_path)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    new_im_names.append(new_im_name)
  return new_im_names


def change_im_names(ori_im_names,  parse_im_name, new_im_name_tmpl):
  """Rename images."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_name in ori_im_names:
    im_name = osp.basename(im_name)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    new_im_names.append(new_im_name)
  return new_im_names


def names2newnames(ori_im_names,  parse_im_name, new_im_name_tmpl):
  """Rename images."""
  new_im_names = change_im_names(ori_im_names,  parse_im_name, new_im_name_tmpl)
  im_names = []
  for i in ori_im_names:
    im_name = osp.basename(i)
    im_names.append(im_name)
  names2new = dict(zip(im_names, new_im_names))
  return names2new



def partition_train_val_set(im_names, parse_im_name,
                            num_val_ids=None, val_prop=None, seed=1):
  """Partition the trainval set into train and val set. 
  Args:
    im_names: trainval image names
    parse_im_name: a function to parse id and camera from image name
    num_val_ids: number of ids for val set. If not set, val_prob is used.
    val_prop: the proportion of validation ids
    seed: the random seed to reproduce the partition results. If not to use, 
      then set to `None`.
  Returns:
    a dict with keys (`train_im_names`, 
                      `val_query_im_names`, 
                      `val_gallery_im_names`)
  """
  np.random.seed(seed)
  # Transform to numpy array for slicing.
  if not isinstance(im_names, np.ndarray):
    im_names = np.array(im_names)
  np.random.shuffle(im_names)
  ids = np.array([parse_im_name(n, 'id') for n in im_names])
  cams = np.array([parse_im_name(n, 'cam') for n in im_names])
  unique_ids = np.unique(ids)
  np.random.shuffle(unique_ids)

  # Query indices and gallery indices
  query_inds = []
  gallery_inds = []

  if num_val_ids is None:
    assert 0 < val_prop < 1
    num_val_ids = int(len(unique_ids) * val_prop)
  num_selected_ids = 0
  for unique_id in unique_ids:
    query_inds_ = []
    # The indices of this id in trainval set.
    inds = np.argwhere(unique_id == ids).flatten()
    # The cams that this id has.
    unique_cams = np.unique(cams[inds])
    # For each cam, select one image for query set.
    for unique_cam in unique_cams:
      query_inds_.append(
        inds[np.argwhere(cams[inds] == unique_cam).flatten()[0]])
    gallery_inds_ = list(set(inds) - set(query_inds_))
    # For each query image, if there is no same-id different-cam images in
    # gallery, put it in gallery.
    for query_ind in query_inds_:
      if len(gallery_inds_) == 0 \
          or len(np.argwhere(cams[gallery_inds_] != cams[query_ind])
                     .flatten()) == 0:
        query_inds_.remove(query_ind)
        gallery_inds_.append(query_ind)
    # If no query image is left, leave this id in train set.
    if len(query_inds_) == 0:
      continue
    query_inds.append(query_inds_)
    gallery_inds.append(gallery_inds_)
    num_selected_ids += 1
    if num_selected_ids >= num_val_ids:
      break

  query_inds = np.hstack(query_inds)
  gallery_inds = np.hstack(gallery_inds)
  val_inds = np.hstack([query_inds, gallery_inds])
  trainval_inds = np.arange(len(im_names))
  train_inds = np.setdiff1d(trainval_inds, val_inds)

  train_inds = np.sort(train_inds)
  query_inds = np.sort(query_inds)
  gallery_inds = np.sort(gallery_inds)

  partitions = dict(train_im_names=im_names[train_inds],
                    val_query_im_names=im_names[query_inds],
                    val_gallery_im_names=im_names[gallery_inds])

  return partitions


def partition_gallery_query_set(im_names, parse_im_name):
  """Partition the galleryquery set into gallery and query set. (For MARS!!!)
  Args:
    im_names: galleryquery image names
    parse_im_name: a function to parse id and camera from image name
  Returns:
    a dict with keys (`gallery_im_names`, 
                      `query_im_names`)
  """
  # Transform to numpy array for slicing.
  if not isinstance(im_names, np.ndarray):
    im_names = np.array(im_names)
  # np.random.shuffle(im_names)
  # ids = np.array([parse_im_name(n, 'id') for n in im_names])
  # cams = np.array([parse_im_name(n, 'cam') for n in im_names])
  # unique_ids = np.unique(ids)
  # np.random.shuffle(unique_ids)

  track_test_info_mat_file = '/GPUFS/nsccgz_ywang_1/MARS/tracks_test_info.mat'
  query_IDX_mat_file = '/GPUFS/nsccgz_ywang_1/MARS/query_IDX.mat'
  track_test = sio.loadmat(track_test_info_mat_file)
  query_IDX = sio.loadmat(query_IDX_mat_file)
  track_test = track_test['track_test_info']
  query_IDX = query_IDX['query_IDX']
  query_idx = query_IDX.squeeze()
  track_query = track_test[query_idx-1,:]
  
  # gallery indices and query indices
  im_num = im_names.shape[0]
  inds = range(im_num)
  gallery_inds = []
  query_inds = []

  k = 0 # the first inds of gallery_inds is inds[k]

  # There are 681089 images in bbox_test.
  # 23380 of 681089, ID = "00-1" means junk images which do not affect retrieval accuracy
  # 147743 of 681089, ID = "0000" means distractors, which negatively affect retrieval accuracy

  # 681089-23380 = 655709
  # 681089-23380-147743 = 681089-171123 = 509966
  if im_num == 655709:
    delete_num = 23380
  else:
    if im_num == 509966:
      delete_num = 171123
  for i in range(query_idx.shape[0]):
    b_num = track_query[i][0] - delete_num
    f_num = track_query[i][1] - delete_num
    gallery_inds = gallery_inds + inds[k:b_num-1]
    query_inds = query_inds + inds[b_num-1:f_num]
    k = f_num
    if k >= im_num:
      break

  gallery_inds = np.hstack(gallery_inds)
  query_inds = np.hstack(query_inds)

  gallery_inds = np.sort(gallery_inds)
  query_inds = np.sort(query_inds)

  print('the first and the last of gallery_inds:   ',gallery_inds[0],gallery_inds[-1])
  print('the first and the last of query_inds:   ',query_inds[0],query_inds[-1])

  print('partitions of gallery and query ...')

  partitions = dict(gallery_im_names=im_names[gallery_inds],
                    query_im_names=im_names[query_inds])

  return partitions


def mess_up_apn(anchor_names, positive_names, negative_names, tolist=1):
  """
  mess up the order of (anchor, positive, negative)
  But retain the corresponding relationship amongs (anchor, positive, negative)
  Args:
    anchor_names: the list or array of anchor names
    positive_names: the list or array of positive names
    negative_names: the list or array of negative names
    tolist: 1 which means return list , else means return array
  Return:
    new_anchor_names, new_positive_names, new_negative_names
  """
  inds = np.arange(len(anchor_names))
  np.random.shuffle(inds)
  if not isinstance(anchor_names, np.ndarray):
    anchor_names = np.array(anchor_names)
  if not isinstance(positive_names, np.ndarray):
    positive_names = np.array(positive_names)
  if not isinstance(negative_names, np.ndarray):
    negative_names = np.array(negative_names)
  if tolist==1:
    new_anchor_names = anchor_names[inds].tolist()
    new_positive_names = positive_names[inds].tolist()
    new_negative_names = negative_names[inds].tolist()
  else:
    new_anchor_names = anchor_names[inds]
    new_positive_names = positive_names[inds]
    new_negative_names = negative_names[inds]
  return new_anchor_names, new_positive_names, new_negative_names


def dist1(feat_a, feat_b):
  """
  Args:
    feat_a: the feature of image_a
    feat_b: the feature of image_b
  Return:
    dist: the number, which means the distance between feat_a and feat_b
  """
  feat_a = torch.FloatTensor(feat_a)
  feat_b = torch.FloatTensor(feat_b)
  dist = torch.norm(feat_a - feat_b, 2)
  return dist


def dist2(feat_a, feat_b):
  """
  Args:
    feat_a: the feature of image_a
    feat_b: a list of features, on a list of images
  Return:
    dist: a list, which means a list of distances between feat_a and each one in feat_b
  """
  dist=[]
  feat_a = torch.FloatTensor(feat_a)
  for i in range(len(feat_b)):
    feat_i = torch.FloatTensor(feat_b[i])
    dist_i = torch.norm(feat_a - feat_i, 2)
    dist.append(dist_i)
  return dist




def hard_sample(af, pf, nf, small=5):
  """
  Args:
    af: the feature of anchor
    pf: the feature of positive
    nf: the feature of negative
    small: choose the five with the smallest distance between d_p and d_n
  Return:
    hard_index: the index of (anchor,positive,negative) which is hard_negative
  """
  d_p = dist1(af, pf)
  d_n = dist2(af, nf)
  d_p_l = []
  for i in range(len(d_n)):
    d_p_l.append(d_p)
  d_p_a = np.array(d_p_l)
  d_n_a = np.array(d_n)
  all = abs(d_n_a-d_p_a)
  index = np.argsort(all)
  n = small
  hard_index = index[0:n]
  return hard_index


