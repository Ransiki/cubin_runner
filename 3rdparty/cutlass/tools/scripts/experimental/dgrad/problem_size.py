#
#
import sys, re, os
from enum import Enum
# import parent directory (tools/scripts)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from conv_network import Network, Layer2D, Resnet50

#
class Mode(Enum):
  Conv = 0
  xCross = 1

#
class ProblemSize:

  def __init__(self, layer):

    self.N = layer.params['n']
    self.H = layer.params['h']
    self.W = layer.params['w']
    self.C = layer.params['c']
    
    self.K = layer.params['k']
    self.R = layer.params['r']
    self.S = layer.params['s']
    
    self.pad_h = layer.params['pad_h']
    self.pad_w = layer.params['pad_w']

    self.stride_h = layer.params['stride_h']
    self.stride_w = layer.params['stride_w']

    self.mode = Mode.xCross

    # sign of p and q increments
    self.sign = -1 if self.mode == Mode.Conv else 1

    # compute and set P and Q
    self.P = ((self.H + self.pad_h * 2 - self.R) // self.stride_h) + 1
    self.Q = ((self.W + self.pad_w * 2 - self.S) // self.stride_w) + 1

    self.NHW = self.N * self.H * self.W

  # overload str operator
  def __str__(self):
    string = "NHWC: (%d, %d, %d, %d)\n"\
             "KRSC: (%d, %d, %d, %d)\n"\
             "NPQK: (%d, %d, %d, %d)\n"\
             "stride: (%d, %d)\n"\
             "padding: (%d, %d)\n"\
             "mode: %s" % \
             (self.N, self.H, self.W, self.C,\
              self.K, self.R, self.S, self.C,\
              self.N, self.P, self.Q, self.K,\
              self.stride_h, self.stride_w,\
              self.pad_h, self.pad_w,
              "Conv" if self.mode == Mode.Conv else "xCross")

    return string

  # check constraints on problem sizes (if any we want enforce for strided dgard)
  def is_valid(self):
    if self.pad_h != self.R // 2 or self.pad_w != self.S // 2: 
      return false

  def set_mode(self, mode):
    self.mode = mode

  # div-mod N, H, W coordinates from NHW linearized index
  def div_mod_nhw(self, idx):
    w = idx % self.W
    residual = idx // self.W
    h = residual % self.H
    n = residual // self.H
    return (n, h, w)

  ############################################################################
  #      Div Mod on N, P, Q for mapping NHW (Working for most sizes so far)
  ############################################################################

  # currently used in CUDA for mapping 
  # (STEP 1) in MMAs 
  # (STEP 4) in epilogue
  def npq(self, idx, start_h, start_w):

    P = (self.H - start_h + self.stride_h - 1) // self.stride_h
    Q = (self.W - start_w + self.stride_w - 1) // self.stride_w

    q = idx % Q
    residual = idx // Q
    p = residual % P
    n = residual // P
    return (n, p, q)
  ############################################################################

  ############################################################################
  #      Div Mod on N, P, Q for mapping NHW (Failed attempts!!!)
  ############################################################################
  def npq_fix_1(self, idx, filter_r, filter_s):

    P = (self.H + self.pad_h - filter_r) // self.stride_h
    Q = (self.W + self.pad_w - filter_s) // self.stride_w

    q = idx % Q
    residual = idx // Q
    p = residual % P
    n = residual // P
    return (n, p, q)


  def npq_fix_2(self, idx, start_h, start_w):

    P = (self.H - start_h) // self.stride_h + 1
    Q = (self.W - start_w) // self.stride_w + 1
    
    q = idx % Q
    residual = idx // Q
    p = residual % P
    n = residual // P
    return (n, p, q)

  def npq_fix_3(self, idx, start_h, start_w, start_r, start_s):

    P = (self.H + 2 * self.pad_h - start_h - (self.R - start_r - 1)) // self.stride_h + 1
    Q = (self.W + 2 * self.pad_h - start_w - (self.S - start_s - 1)) // self.stride_w + 1
    
    q = idx % Q
    residual = idx // Q
    p = residual % P
    n = residual // P
    return (n, p, q)
  ############################################################################

#
# Add Strided layers
#
StridedLayers = Network("StridedLayers")

StridedLayers.add_layer(Layer2D(
  [1, 4, 4, 8], \
  [8, 3, 3, 8], \
  [0, 0, 0, 0], \
  [3, 3], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 4, 4, 8], \
  [8, 3, 3, 8], \
  [1, 1, 1, 1], \
  [3, 3], \
  [1, 1], \
))

StridedLayers.add_layer(Layer2D(
  [1, 55, 55, 8], \
  [8, 1, 1, 8], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 8, 8, 8], \
  [8, 5, 5, 8], \
  [2, 2, 2, 2], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 224, 224, 8], \
  [8, 7, 7, 8], \
  [3, 3, 3, 3], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 5, 5, 8], \
  [8, 3, 3, 8], \
  [1, 1, 1, 1], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 5, 5, 8], \
  [8, 1, 1, 8], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 5, 5, 8], \
  [8, 3, 3, 8], \
  [1, 1, 1, 1], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 8, 8, 8], \
  [8, 3, 3, 8], \
  [1, 1, 1, 1], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 8, 8, 8], \
  [8, 2, 2, 8], \
  [1, 1, 1, 1], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 8, 8, 8], \
  [8, 1, 1, 8], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1], \
));

StridedLayers.add_layer(Layer2D(
  [1, 56, 56, 8], \
  [8, 1, 1, 8], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1], \
));
