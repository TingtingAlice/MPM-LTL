"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np
import torch


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  elif type == 'euclidean':
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist

def compute_dist2(array1, array2):
  """Compute the euclidean distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
  Returns:
    numpy array with shape [m1, m2]
  """
  a1 = torch.FloatTensor(array1)
  a2 = torch.FloatTensor(array2)
  dist = []
  for i in range(array1.shape[0]):
    a_i = a1[i]
    dist_i = torch.norm(a_i - a2, 2, 1)
    dist_i = np.array(dist_i)
    dist.append(dist_i)
  dist = np.array(dist)
  return dist


def compute_dist_triplet(feat_list1, feat_list2):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert feat_list1.size() == feat_list2.size()
  eps = 1e-4 / feat_list1.size(1)
  diff = torch.abs(feat_list1 - feat_list2)
  out = torch.pow(diff, self.norm).sum(dim=1)
  return torch.pow(out + eps, 1. / self.norm)