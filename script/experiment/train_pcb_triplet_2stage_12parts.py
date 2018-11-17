# -*- coding: utf-8 -*
from __future__ import print_function

import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from bpm.dataset import create_dataset,create_dataset_tri
# from bpm.model.PCBModel import PCBModel as Model
from bpm.model.PCB_ColModel import PCBModel as Model

from bpm.utils.utils import time_str
from bpm.utils.utils import str2bool
from bpm.utils.utils import may_set_mode
from bpm.utils.utils import load_state_dict
from bpm.utils.utils import load_ckpt
from bpm.utils.utils import save_ckpt
from bpm.utils.utils import set_devices
from bpm.utils.utils import AverageMeter
from bpm.utils.utils import to_scalar
from bpm.utils.utils import ReDirectSTD
from bpm.utils.utils import set_seed
from bpm.utils.utils import adjust_lr_staircase,PairwiseDistance
# from bpm.utils.utils import display_triplet_distance,display_triplet_distance_test
from torch.autograd import Function
from bpm.utils.distance import compute_dist_triplet
from torch.autograd import Variable


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,1,2,3))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--pcb_dataset', type=str, default='market1501',
                        choices=['market_png_4_1','market_png','market30_retain_pixel1','market30_retain_pixel2','market30_retain_pixel4',\
                        'market30_retain_pixel5','market30_retain_pixel6',\
                        'market30_retain_pixel7','market30_retain_pixel8','market30_retain_pixel9','market30_retain_pixel10',\
                        'market30_retain_pixel3','cuhk33_retain_3','market30_retain_pixel0_3_1','mars30_retain_pixel7','mars32_retain_pixel7','mars33_retain_pixel7','market30_retain_2','market30_retain_3','market30_retain_pixel0_2','market30_retain_pixel0_3',\
                        'market30_retain_pixel0','mars_oldmask_retain','mars','mars20','mars22','mars23','mars30','mars32','mars33','market',\
                        'cuhk20','cuhk22','cuhk23','cuhk20_retain','cuhk22_retain','cuhk23_retain','cuhk30','cuhk32','cuhk33',\
                        'cuhk30_retain','cuhk32_retain','cuhk33_retain','cuhk40','cuhk42','cuhk43','cuhk40_retain','cuhk42_retain',\
                        'cuhk43_retain','market1501','market_combined','market23','market22', 'market20','market20_retain','market22_retain',\
                        'market23_retain', 'market30','market32','market33','market30_retain','market32_retain','market33_retain',\
                        'market40','market42','market43','market40_retain','market42_retain','market43_retain','market_oldmask',\
                        'market_oldmask_retain','market_trans','market_png'])
    parser.add_argument('--triplet_dataset', type=str, default='market1501',
                        choices=['market_png_4_1','market_png','market30_retain_pixel3_rand_1','market30_retain_pixel1_4_1',\
                        'market30_retain_pixel2_4_1','market30_retain_pixel4_4_1','market30_retain_pixel5_4_1',\
                        'market30_retain_pixel6_4_1','market30_retain_pixel7_4_1','market30_retain_pixel8_4_1','market30_retain_pixel9_4_1',\
                        'market30_retain_pixel10_4_1','market30_retain_rand_1','market30_retain_pixel3_3_1','market30_retain_pixel3_4_1',\
                        'market30_retain_pixel3_5_3','market30_retain_pixel3_rand_1',\
                        'cuhk33_retain_3_1','cuhk33_retain_4','cuhk33_retain_4_1','cuhk33_retain_5','cuhk33_retain_5_3','cuhk33_retain_5_6',\
                        'market30_retain_3_1','market30_retain_4','market30_retain_4_1','market30_retain_5',\
                        'market30_retain_5_3','market30_retain_5_6','market33_retain_5','market33_retain_5_3','market33_retain_5_6',\
                        'market33_retain_3','market33_retain_3_1','market33_retain_4','market33_retain_4_1',\
                        'market30_retain_pixel0_4_1','market30_retain_pixel0_5_6','market30_retain_pixel0_5_3','market30_retain_pixel0_5',\
                        'market30_retain_pixel0_4_5','cuhk33_retain_3','market30_retain_pixel0_3_1','mars30_retain_pixel7','mars32_retain_pixel7',\
                        'mars33_retain_pixel7','market30_retain_2','market30_retain_3','market30_retain_pixel0_2','market30_retain_pixel0_3',\
                        'market30_retain_pixel0','mars_oldmask_retain','mars','mars20','mars22','mars23','mars30','mars32','mars33','market',\
                        'cuhk20','cuhk22','cuhk23','cuhk20_retain','cuhk22_retain','cuhk23_retain','cuhk30','cuhk32','cuhk33',\
                        'cuhk30_retain','cuhk32_retain','cuhk33_retain','cuhk40','cuhk42','cuhk43','cuhk40_retain','cuhk42_retain',\
                        'cuhk43_retain','market1501','market_combined','market23','market22', 'market20','market20_retain','market22_retain',\
                        'market23_retain', 'market30','market32','market33','market30_retain','market32_retain','market33_retain',\
                        'market40','market42','market43','market40_retain','market42_retain','market43_retain','market_oldmask',\
                        'market_oldmask_retain','market_trans','market_png'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    parser.add_argument('--resize_h_w', type=eval, default=(384, 128))
    # These several only for training set
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--mirror', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--triplet_batch_size', type=int, default=192)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=20)
    parser.add_argument('--epochs_per_val', type=int, default=1)

    parser.add_argument('--last_conv_stride', type=int, default=1, choices=[1, 2])
    # When the stride is changed to 1, we can compensate for the receptive field
    # using dilated convolution. However, experiments show dilated convolution is useless.
    parser.add_argument('--last_conv_dilation', type=int, default=1, choices=[1, 2])
    parser.add_argument('--num_stripes', type=int, default=6)
    parser.add_argument('--num_cols', type=int, default=1)
    parser.add_argument('--local_conv_out_channels', type=int, default=256)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--only_triplet', type=str2bool, default=False)
    parser.add_argument('--only_all', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--new_params_lr', type=float, default=0.1)
    parser.add_argument('--finetuned_params_lr', type=float, default=0.01)
    parser.add_argument('--triplet_finetuned_params_lr', type=float, default=0.001)
    parser.add_argument('--all_base_finetuned_params_lr', type=float, default=0.001)
    parser.add_argument('--all_new_finetuned_params_lr', type=float, default=0.01)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(41,))
    parser.add_argument('--all_staircase_decay_at_epochs',
                        type=eval, default=(25,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--triplet_staircase_decay_multiply_factor',
                        type=float, default=1)
    parser.add_argument('--total_epochs', type=int, default=5)
    parser.add_argument('--triplet_epochs', type=int, default=5)
    parser.add_argument('--pcb_epochs', type=int, default=60)
    parser.add_argument('--margin', type=float, default=0.5, metavar='MARGIN')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MOMENTTUM')
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    args = parser.parse_args()
    #lr
    self.triplet_finetuned_params_lr = args.triplet_finetuned_params_lr
    self.all_base_finetuned_params_lr = args.all_base_finetuned_params_lr
    self.all_new_finetuned_params_lr = args.all_new_finetuned_params_lr

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    #parts number
    self.parts_num = args.num_cols*args.num_stripes

    # margin
    self.margin = args.margin

    # If you want to make your results exactly reproducible, you have
    # to fix a random seed.
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to make your results exactly reproducible, you have
    # to also set num of threads to 1 during training.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.pcb_dataset = args.pcb_dataset
    self.triplet_dataset = args.triplet_dataset
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    # self.im_mean = [0.486, 0.459, 0.408,0.5]
    # self.im_std = [0.229, 0.224, 0.225,0.5]
    self.im_mean = None
    self.im_std = None

    self.train_mirror_type = 'random' if args.mirror else None
    self.pcb_train_batch_size = args.batch_size
    self.train_batch_size = args.triplet_batch_size
    self.train_final_batch = False
    #modify the train_shuffle and keep the corresponding relation
    self.train_shuffle = False
    self.pcb_train_shuffle = True

    self.test_mirror_type = None
    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_shuffle = False
    self.pcb_test_shuffle = True

    triplet_dataset_kwargs = dict(
      name=self.triplet_dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)
    pcb_dataset_kwargs = dict(
      name=self.pcb_dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)
    """
    pcb dataset
    """
    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.pcb_train_set_kwargs = dict(
      part=self.trainset_part,
      batch_size=self.pcb_train_batch_size,
      final_batch=self.train_final_batch,
      shuffle=self.pcb_train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.pcb_train_set_kwargs.update(pcb_dataset_kwargs)

    """
    all dataset
    """
    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.all_train_set_kwargs = dict(
      part=self.trainset_part,
      batch_size=self.train_batch_size,
      final_batch=self.train_final_batch,
      shuffle=self.pcb_train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.all_train_set_kwargs.update(pcb_dataset_kwargs)


    """
    triplet dataset
    """
    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_anchor_kwargs = dict(
      flag= 'anchor',
      part=self.trainset_part,
      batch_size=self.train_batch_size,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_anchor_kwargs.update(triplet_dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_positive_kwargs = dict(
      flag= 'positive',
      part=self.trainset_part,
      batch_size=self.train_batch_size,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_positive_kwargs.update(triplet_dataset_kwargs)
    
    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_negative_kwargs = dict(
      flag= 'negative',
      part=self.trainset_part,
      batch_size=self.train_batch_size,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_negative_kwargs.update(triplet_dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.val_set_kwargs = dict(
      part='val',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.val_set_kwargs.update(triplet_dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(triplet_dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    # The last block of ResNet has stride 2. We can set the stride to 1 so that
    # the spatial resolution before global pooling is doubled.
    self.last_conv_stride = args.last_conv_stride
    # When the stride is changed to 1, we can compensate for the receptive field
    # using dilated convolution. However, experiments show dilated convolution is useless.
    self.last_conv_dilation = args.last_conv_dilation
    # Number of stripes (parts)
    self.num_stripes = args.num_stripes
    self.num_cols = args.num_cols
    # Output channel of 1x1 conv
    self.local_conv_out_channels = args.local_conv_out_channels

    #############
    # Training  #
    #############

    self.momentum = args.momentum
    self.weight_decay = args.weight_decay

    # Initial learning rate
    self.new_params_lr = args.new_params_lr
    self.finetuned_params_lr = args.finetuned_params_lr
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    self.triplet_staircase_decay_multiply_factor = args.triplet_staircase_decay_multiply_factor
    self.all_staircase_decay_at_epochs = args.all_staircase_decay_at_epochs
    # Number of epochs to train
    self.total_epochs = args.total_epochs
    self.triplet_epochs = args.triplet_epochs
    self.pcb_epochs = args.pcb_epochs

    # How often (in epochs) to test on val set.
    self.epochs_per_val = args.epochs_per_val

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.steps_per_log = args.steps_per_log

    # Only test and without training.
    self.only_test = args.only_test

    #Only triplet without pcb training ,load pcb_ckpt file
    self.only_triplet = args.only_triplet 
    self.only_all = args.only_all

    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.pcb_dataset),
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.pcb_ckpt_file = osp.join(self.exp_dir, 'pcb_ckpt.pth')
    self.triplet_ckpt_file = osp.join(self.exp_dir, 'triplet_ckpt.pth')
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()

    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    try:
      local_feat_list, logits_list = self.model(ims)
    except:
      local_feat_list = self.model(ims)
    feat = [lf.data.cpu().numpy() for lf in local_feat_list]
    feat = np.concatenate(feat, axis=1)

    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return feat

class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge) 
        return loss

def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)


  #####################################################
  # Dataset #
  # create train dataset for pcb loss#
  #####################################################
  print('start to create pcb dataset.....')
  train_set = create_dataset(**cfg.pcb_train_set_kwargs)
  print('train_set shape:{}'.format(len(train_set.im_names)))
  num_classes = len(train_set.ids2labels)

  #####################################################
  # Dataset #
  # create val_set test_set for pcb loss#
  #####################################################
  # The combined dataset does not provide val set currently.
  val_set = None if cfg.pcb_dataset == 'combined' else create_dataset(**cfg.val_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.pcb_dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.pcb_dataset)

  ########
  # Test #
  ########

  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        load_ckpt(modules_optims, cfg.ckpt_file)

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      test_set.eval(
        normalize_feat=True,
        verbose=True)

  def validate():
    if val_set.extract_feat_func is None:
      val_set.set_feat_func(ExtractFeature(model_w, TVT))
    print('\n===== Test on validation set =====\n')
    mAP, cmc_scores, _, _ = val_set.eval(
      normalize_feat=True,
      to_re_rank=False,
      verbose=True)
    print()
    return mAP, cmc_scores[0]

  #############################################################################
                       # train pcb model for 60eps #
  #############################################################################
  
  if not cfg.only_triplet:
    model = Model(
    last_conv_stride=cfg.last_conv_stride,
    num_stripes=cfg.num_stripes,
    num_cols=cfg.num_cols,
    local_conv_out_channels=cfg.local_conv_out_channels,
    num_classes=num_classes)

    # Model wrapper
    model_w = DataParallel(model)


    #############################
    # Criteria and Optimizers   #
    #############################

    criterion = torch.nn.CrossEntropyLoss()

    # To finetune from ImageNet weights
    finetuned_params = list(model.base.parameters())
    # To train from scratch
    new_params = [p for n, p in model.named_parameters()
                if not n.startswith('base.')]
    param_groups = [{'params': finetuned_params, 'lr': cfg.finetuned_params_lr},
                  {'params': new_params, 'lr': cfg.new_params_lr}]
    optimizer = optim.SGD(
      param_groups,
      momentum=cfg.momentum,
      weight_decay=cfg.weight_decay)

      # Bind them together just to save some codes in the following usage.
  
    modules_optims = [model, optimizer]

    ################################
    # May Resume Models and Optims #
    ################################

    if cfg.resume:
      resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

    # May Transfer Models and Optims to Specified Device. Transferring optimizer
    # is to cope with the case when you load the checkpoint to a new device.
    TMO(modules_optims)

    if cfg.only_test:
      test(load_model_weight=True)
      return


    print('#####################Begin to train pcb model##############################')
    ############
    # Training #
    ############

    start_ep = resume_ep if cfg.resume else 0
    for ep in range(start_ep, cfg.pcb_epochs):

      # Adjust Learning Rate
      adjust_lr_staircase(
        optimizer.param_groups,
        [cfg.finetuned_params_lr, cfg.new_params_lr],
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiply_factor)
      # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
      may_set_mode(modules_optims, 'train')

      # For recording loss
      loss_meter = AverageMeter()

      ep_st = time.time()
      step = 0
      epoch_done = False
      while not epoch_done:

        step += 1
        step_st = time.time()

        ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

        ims_var = Variable(TVT(torch.from_numpy(ims).float()))
        labels_var = Variable(TVT(torch.from_numpy(labels).long()))

        _, logits_list = model_w(ims_var)
        loss = torch.sum(
          torch.cat([criterion(logits, labels_var) for logits in logits_list]))
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del logits_list,ims,im_names,labels

        ############
        # Step Log #
        ############

        loss_meter.update(to_scalar(loss))

        if step % cfg.steps_per_log == 0:
          log = '\tStep {}/Ep {}, {:.2f}s, loss {:.4f}'.format(
            step, ep + 1, time.time() - step_st, loss_meter.val)
          print(log)

      ################
      # adjust the lr#
      ################
      # scheduler.step(loss_meter.avg)
      #############
      # Epoch Log #
      #############
      
      log = 'Ep {}, {:.2f}s, loss {:.4f}'.format(
        ep + 1, time.time() - ep_st, loss_meter.avg)
      print(log)


      ##########################
      # Test on Validation Set #
      ##########################

      mAP, Rank1 = 0, 0
      if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
        mAP, Rank1 = validate()

      # Log to TensorBoard

      if cfg.log_to_file:
        if writer is None:
          writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
        writer.add_scalars(
          'val scores',
          dict(mAP=mAP,
              Rank1=Rank1),
          ep)
        writer.add_scalars(
          'loss',
          dict(loss=loss_meter.avg, ),
          ep)

      # save ckpt
      if cfg.log_to_file:
        save_ckpt(modules_optims, ep + 1, 0, cfg.pcb_ckpt_file)

    ################
    # Test the pcb #
    ################

    test(load_model_weight=False)

  #####################################################
  # Dataset #
  # create train_set val_set test_set for triplet loss#
  # todo #
  #####################################################
  print('start to create triplet_all dataset.....')
  train_set_anchor = create_dataset_tri(**cfg.train_set_anchor_kwargs)
  print('train_anchor_im_names:{}'.format(len(train_set_anchor.get_im_names())))
  train_set_positive = create_dataset_tri(**cfg.train_set_positive_kwargs)
  print('train_positive_im_names:{}'.format(len(train_set_positive.get_im_names())))
  train_set_negative = create_dataset_tri(**cfg.train_set_negative_kwargs)
  print('train_negative_im_names:{}'.format(len(train_set_negative.get_im_names())))
  num_classes = len(train_set_anchor.ids2labels)
  print('finish creating....num_classes:{}\n '.format(num_classes))
    
  
  if not cfg.only_all:

    print('come into triplet*********************************************')

    
    ##################################################################################
                                # Models for triplet 5 eps #
    ###################################################################################
    model = Model(
      last_conv_stride=cfg.last_conv_stride,
      num_stripes=cfg.num_stripes,
      num_cols=cfg.num_cols,
      local_conv_out_channels=cfg.local_conv_out_channels,
      num_classes=num_classes
    )

    #############################
    # Criteria and Optimizers   #
    #############################

    criterion = torch.nn.CrossEntropyLoss()

    #load the checkpoint
    if osp.isfile(cfg.pcb_ckpt_file):
      map_location = (lambda storage, loc: storage)
      sd = torch.load(cfg.pcb_ckpt_file, map_location=map_location)
      model_dict = model.state_dict()
      sd_load = {k: v for k, v in (sd['state_dicts'][0]).items() if k in model_dict}
      model_dict.update(sd_load)
      model.load_state_dict(model_dict)
    
    # # Optimizer
    #   if hasattr(model.module, 'base'):
    #       base_param_ids = set(map(id, model.base.parameters()))
    #       conv_list_ids = set(map(id,model.local_conv_list.parameters()))
    #       fc_list_ids = set(map(id,model.fc_list.parameters()))
    #       new_params = [p for p in model.parameters() if
    #                     id(p) not in base_param_ids and id(p) not in fc_list_ids]
    #       param_groups = [{'params': new_params, 'lr': 1.0}]
    #   else:
    #       param_groups = model.parameters()
      

    # To finetune from ImageNet weights
    # finetuned_params = list(model.base.parameters())
    # To train from scratch
    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base.') and not n.startswith('fc_list.')]
    param_groups = [{'params': new_params, 'lr': cfg.new_params_lr}]
    optimizer = optim.SGD(
      param_groups,
      momentum=cfg.momentum,
      weight_decay=cfg.weight_decay)


    # Model wrapper
    model_w = DataParallel(model)
    # Bind them together just to save some codes in the following usage.
    
    modules_optims = [model, optimizer]

    TMO(modules_optims)

  ############
    # Training #
    ############
    print('#####################Begin to train triplet model##############################')
    start_ep = resume_ep if cfg.resume else 0
    triplet_st = time.time()
    for ep in range(start_ep, cfg.triplet_epochs):

      # Adjust Learning Rate
      adjust_lr_staircase(
        optimizer.param_groups,
        [cfg.triplet_finetuned_params_lr],
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.triplet_staircase_decay_multiply_factor)
      # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
      may_set_mode(modules_optims, 'train')

      # For recording loss
      loss_meter = AverageMeter()

      ep_st = time.time()
      step = 0
      epoch_done = False
      while not epoch_done:

        step += 1
        step_st = time.time()

        ims_a, im_names_a, labels_a, mirrored_a, epoch_done = train_set_anchor.next_batch()
        ims_p, im_names_p, labels_p, mirrored_p, epoch_done = train_set_positive.next_batch()
        ims_n, im_names_n, labels_n, mirrored_n, epoch_done = train_set_negative.next_batch()

        ims_var_a,ims_var_p,ims_var_n = Variable(TVT(torch.from_numpy(ims_a).float())),Variable(TVT(torch.from_numpy(ims_p).float())),\
                                      Variable(TVT(torch.from_numpy(ims_n).float()))
        labels_a_var = Variable(TVT(torch.from_numpy(labels_a).long()))
        labels_n_var = Variable(TVT(torch.from_numpy(labels_n).long()))

        local_feat_list_a, logits_list_a = model_w(ims_var_a)
        local_feat_list_p, logits_list_p = model_w(ims_var_p)
        local_feat_list_n, logits_list_n = model_w(ims_var_n)

        loss_triplet = []
        #print('shape of local_feat:{}  {}'.format(len(local_feat_list_a),local_feat_list_a[0].shape))
        # print('Ep{}: '.format(ep+1))
        loss_local = Variable(torch.Tensor([0]))
        for i in range(cfg.parts_num):
          #print(i)
          loss_triplet.append(TripletMarginLoss(cfg.margin).forward(local_feat_list_a[i], local_feat_list_p[i], local_feat_list_n[i]))
        # print('the {}th local loss: {}'.format(i,loss_triplet[i]))
          # loss_local = loss_local+loss_triplet[i].data

        if cfg.parts_num == 6:
          #get the local loss
          #loss_local_all =0.1* loss_triplet[0]+0.2*loss_triplet[1]+0.2*loss_triplet[2]+0.2*loss_triplet[3]+0.2*loss_triplet[4]+0.1*loss_triplet[5]
          loss_local_all =loss_triplet[0]+loss_triplet[1]+loss_triplet[2]+loss_triplet[3]+loss_triplet[4]+loss_triplet[5]
                          
        elif cfg.parts_num == 12:
          loss_local_all =loss_triplet[0]+loss_triplet[1]+loss_triplet[2]+loss_triplet[3]+loss_triplet[4]+loss_triplet[5]+\
          loss_triplet[6]+loss_triplet[7]+loss_triplet[8]+loss_triplet[9]+loss_triplet[10]+loss_triplet[11]
                            
        elif cfg.parts_num == 18:
          loss_local_all =loss_triplet[0]+loss_triplet[1]+loss_triplet[2]+loss_triplet[3]+loss_triplet[4]+loss_triplet[5]+\
          loss_triplet[6]+loss_triplet[7]+loss_triplet[8]+loss_triplet[9]+loss_triplet[10]+loss_triplet[11]+\
          loss_triplet[12]+loss_triplet[13]+loss_triplet[14]+loss_triplet[15]+loss_triplet[16]+loss_triplet[17]

        elif cfg.parts_num == 3:
          loss_local_all =loss_triplet[0]+loss_triplet[1]+loss_triplet[2]
                            
        
        loss = torch.div(loss_local_all,cfg.parts_num)
        # loss = torch.div(torch.sum(local for local in loss_triplet),cfg.parts_num)
        print('loss:{}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # del local_feat_list_a,local_feat_list_p,local_feat_list_n,logits_list_a,logits_list_p,logits_list_n
        ############
        # Step Log #
        ############

        loss_meter.update(to_scalar(loss))

        if step % cfg.steps_per_log == 0:
          log = '\tStep {}/Ep {}, {:.2f}s, loss {:.4f}'.format(
            step, ep + 1, time.time() - step_st, loss_meter.val)
          print(log)

      ################
      # adjust the lr by torch function#
      ################
      # scheduler.step(loss_meter.avg)
      #############
      # Epoch Log #
      #############
      
      log = 'Ep {}, {:.2f}s, loss {:.4f}'.format(
        ep  +1, time.time() - ep_st, loss_meter.avg)
      print(log)

      ##############
      # RPP module #             
      ##############

      ##########################
      # Test on Validation Set #
      ##########################

      mAP, Rank1 = 0, 0
      if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
        mAP, Rank1 = validate()

      # Log to TensorBoard

      if cfg.log_to_file:
        if writer is None:
          writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
        writer.add_scalars(
          'val scores',
          dict(mAP=mAP,
              Rank1=Rank1),
          cfg.pcb_epochs +ep)
        writer.add_scalars(
          'loss',
          dict(loss=loss_meter.avg, ),
          ep+cfg.pcb_epochs )

      # save ckpt
      if cfg.log_to_file:
        save_ckpt(modules_optims, ep +  cfg.pcb_epochs +1, 0, cfg.triplet_ckpt_file)
    
    triplet_time = time.time() - triplet_st

    ########
    # Test #
    ########

    test(load_model_weight=False)

  print('come into all train********************************************88')
##########################################################################
                      # train all for 5 eps #
###########################################################################
  model = Model(
    last_conv_stride=cfg.last_conv_stride,
    num_stripes=cfg.num_stripes,
    num_cols=cfg.num_cols,
    local_conv_out_channels=cfg.local_conv_out_channels,
    num_classes=num_classes
  )

  #####################################################
  # Dataset #
  # create train dataset for pcb loss#
  #####################################################
  print('start to create all dataset.....')
  all_train_set = create_dataset(**cfg.all_train_set_kwargs)
  print('train_set shape:{}'.format(len(all_train_set.im_names)))
  num_classes = len(train_set.ids2labels)
  #############################
  # Criteria and Optimizers   #
  #############################

  criterion = torch.nn.CrossEntropyLoss()

  #load the checkpoint
  if osp.isfile(cfg.triplet_ckpt_file):
    map_location = (lambda storage, loc: storage)
    sd = torch.load(cfg.triplet_ckpt_file, map_location=map_location)
    model_dict = model.state_dict()
    sd_load = {k: v for k, v in (sd['state_dicts'][0]).items() if k in model_dict}
    model_dict.update(sd_load)
    model.load_state_dict(model_dict)
    

  # To finetune from ImageNet weights
  finetuned_params = list(model.base.parameters())
  # To train from scratch
  # new_params = [p for n, p in model.named_parameters()
  #               if not n.startswith('base.')]
  new_params = [p for n, p in model.named_parameters()
                if not n.startswith('base.') and not n.startswith('local_conv_list.')]
  param_groups = [{'params': finetuned_params, 'lr': cfg.finetuned_params_lr*0.1},
                  {'params': new_params, 'lr': cfg.finetuned_params_lr}]
  optimizer = optim.SGD(
    param_groups,
    momentum=cfg.momentum,
    weight_decay=cfg.weight_decay)


  # Model wrapper
  model_w = DataParallel(model)
  # Bind them together just to save some codes in the following usage.
  
  modules_optims = [model, optimizer]

  TMO(modules_optims)

 ############
  # Training #
  ############
  print('#####################Begin to train all model##############################')
  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    adjust_lr_staircase(
      optimizer.param_groups,
      [cfg.all_base_finetuned_params_lr,cfg.all_new_finetuned_params_lr],
      ep + 1,
      cfg.all_staircase_decay_at_epochs,
      cfg.staircase_decay_multiply_factor)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    may_set_mode(modules_optims, 'train')

    # For recording loss
    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      # step += 1
      # step_st = time.time()

      # ims_a, im_names_a, labels_a, mirrored_a, epoch_done = train_set_anchor.next_batch()
      # ims_p, im_names_p, labels_p, mirrored_p, epoch_done = train_set_positive.next_batch()
      # ims_n, im_names_n, labels_n, mirrored_n, epoch_done = train_set_negative.next_batch()

      # ims_var_a,ims_var_p,ims_var_n = Variable(TVT(torch.from_numpy(ims_a).float())),Variable(TVT(torch.from_numpy(ims_p).float())),\
      #                               Variable(TVT(torch.from_numpy(ims_n).float()))
      # labels_a_var = Variable(TVT(torch.from_numpy(labels_a).long()))
      # labels_n_var = Variable(TVT(torch.from_numpy(labels_n).long()))

      # local_feat_list_a, logits_list_a = model_w(ims_var_a)
      # local_feat_list_p, logits_list_p = model_w(ims_var_p)
      # local_feat_list_n, logits_list_n = model_w(ims_var_n)

      # loss_triplet = []
      # #print('shape of local_feat:{}  {}'.format(len(local_feat_list_a),local_feat_list_a[0].shape))
      # # print('Ep{}: '.format(ep+1))
      # loss_local = Variable(torch.Tensor([0]))

      
      # for i in range(cfg.parts_num):
      #   #print(i)
      #   loss_triplet.append(TripletMarginLoss(cfg.margin).forward(local_feat_list_a[i], local_feat_list_p[i], local_feat_list_n[i]))
      #  # print('the {}th local loss: {}'.format(i,loss_triplet[i]))
      #   # loss_local = loss_local+loss_triplet[i].data

      # if cfg.parts_num == 6:
      #     #get the local loss
      #     #loss_local_all =0.1* loss_triplet[0]+0.2*loss_triplet[1]+0.2*loss_triplet[2]+0.2*loss_triplet[3]+0.2*loss_triplet[4]+0.1*loss_triplet[5]
      #     loss_local_all =loss_triplet[0]+loss_triplet[1]+loss_triplet[2]+loss_triplet[3]+loss_triplet[4]+loss_triplet[5]
                          
      # elif cfg.parts_num == 12:
      #     loss_local_all =loss_triplet[0]+loss_triplet[1]+loss_triplet[2]+loss_triplet[3]+loss_triplet[4]+loss_triplet[5]+\
      #     loss_triplet[6]+loss_triplet[7]+loss_triplet[8]+loss_triplet[9]+loss_triplet[10]+loss_triplet[11]
                            
      # elif cfg.parts_num == 18:
      #     loss_local_all =loss_triplet[0]+loss_triplet[1]+loss_triplet[2]+loss_triplet[3]+loss_triplet[4]+loss_triplet[5]+\
      #     loss_triplet[6]+loss_triplet[7]+loss_triplet[8]+loss_triplet[9]+loss_triplet[10]+loss_triplet[11]+\
      #     loss_triplet[12]+loss_triplet[13]+loss_triplet[14]+loss_triplet[15]+loss_triplet[16]+loss_triplet[17]
        
                        
      # loss_local = torch.div(loss_local_all,cfg.parts_num)
      # # print('for loss_local:{}'.format(loss_local))
      # #get the id loss
      # loss_id_a = torch.sum(
      #   torch.cat([criterion(logits, labels_a_var) for logits in logits_list_a]))
      # loss_id_p = torch.sum(
      #   torch.cat([criterion(logits, labels_a_var) for logits in logits_list_p]))
      # loss_id_n = torch.sum(
      #   torch.cat([criterion(logits, labels_n_var) for logits in logits_list_n]))
      # """
      #   get the id using the whole feature
      # """
      # # loss_id_a = criterion(logits_list_a, labels_a_var) 
      # # loss_id_p = criterion(logits_list_a, labels_a_var) 
      # # loss_id_n = criterion(logits_list_n, labels_n_var) 

      # loss_id = torch.div(loss_id_a+loss_id_n+loss_id_p,3) 
      # # print('for id loss：{}'.format(loss_id))

      
      # loss = loss_id+5*loss_local
      # print('loss:{}'.format(loss))
      # optimizer.zero_grad()
      # loss.backward()
      # optimizer.step()

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = all_train_set.next_batch()

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_var = Variable(TVT(torch.from_numpy(labels).long()))

      _, logits_list = model_w(ims_var)
      loss = torch.sum(
        torch.cat([criterion(logits, labels_var) for logits in logits_list]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ############
      # Step Log #
      ############

      loss_meter.update(to_scalar(loss))

      if step % cfg.steps_per_log == 0:
        log = '\tStep {}/Ep {}, {:.2f}s, loss {:.4f}'.format(
          step, ep + 1, time.time() - step_st, loss_meter.val)
        print(log)

    ################
    # adjust the lr by torch function#
    ################
    # scheduler.step(loss_meter.avg)
    #############
    # Epoch Log #
    #############
    
    log = 'Ep {}, {:.2f}s, loss {:.4f}'.format(
      ep  +1, time.time() - ep_st, loss_meter.avg)
    print(log)

    ##############
    # RPP module #             
    ##############

    ##########################
    # Test on Validation Set #
    ##########################

    mAP, Rank1 = 0, 0
    if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
      mAP, Rank1 = validate()

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(mAP=mAP,
             Rank1=Rank1),
         cfg.pcb_epochs +cfg.triplet_epochs+ep)
      writer.add_scalars(
        'loss',
        dict(loss=loss_meter.avg, ),
        ep+cfg.pcb_epochs  +cfg.triplet_epochs)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep +  cfg.pcb_epochs +1 +cfg.triplet_epochs, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  test(load_model_weight=False)
  print('over**************************************************')
  print('{} spends {} s'.format(cfg.triplet_dataset,triplet_time))

if __name__ == '__main__':
  main()
