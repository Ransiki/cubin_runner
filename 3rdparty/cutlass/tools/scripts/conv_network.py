# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import csv
import os
import subprocess
import argparse
import re
from collections import OrderedDict

#################################################################################################
#
class Network:

  #
  def __init__(self, name):
    self.name = name
    self.layers = []

  def add_layer(self, layer):
    
    # set network layer_id if not specified
    if layer.id == -1:
      if not len(self.layers):
        # set first layer with layer_id = 0
        layer.id = 0
      else:
        # set all subsequent layers with one more the than the last layer
        layer.id = self.layers[-1].id + 1

    layer.network = self.name

    self.layers.append(layer)

  def __str__(self):
  	return "Network: %s\n %s" % (self.name, "\n".join(str(layer) for layer in self.layers))

# 2D convolution layers
class Layer2D:

  #
  def __init__(self, input, filter, padding, stride, dilation, id=-1, network="unknown"):

    self.params = OrderedDict([
      ('n', input[0]),
      ('h', input[1]),      
      ('w', input[2]),
      ('c', input[3]),

      ('k', filter[0]),
      ('r', filter[1]),
      ('s', filter[2]),

      ('pad_h', padding[0]),
      ('pad_w', padding[2]),

      ('stride_h', stride[0]),
      ('stride_w', stride[1]),

      ('dilation_h', dilation[0]),
      ('dilation_w', dilation[1]),
    ])

    self.id = id
    self.network = network

  # returns profiler command for the layer
  def profiler_cmd(self, batch_size=1):
    
    cmdline_layer = " ".join(str("--%s=%d" % (key, value)) for (key, value) in self.params.items())

    if batch_size > 1:
      cmdline_layer = re.sub(r'--n=\d+', "".join(['--n=', str(batch_size)]), cmdline_layer.rstrip())

    return cmdline_layer

  def __str__(self):
    layer_details = \
      " %s" % (self.network) +\
      " layer id: %d," % (self.id)+\
      " input: (n=%d,h=%d,w=%d,c=%d)," % (self.params['n'], self.params['h'], self.params['w'], self.params['c']) +\
      " filter: (k=%d,r=%d,s=%d,c=%d)," % (self.params['k'], self.params['r'], self.params['s'], self.params['c']) +\
      " padding: (h=%d,w=%d)," % (self.params['pad_h'], self.params['pad_w']) +\
      " stride: (h=%d,w=%d)," % (self.params['stride_h'], self.params['stride_w']) +\
      " dilation: (h=%d,w=%d)," % (self.params['dilation_h'], self.params['dilation_w'])
    return layer_details

  # returns True if the layer is strided layer
  def is_strided(self):
    if self.params['stride_h'] != 1 or self.params['stride_w'] != 1:
      return True
    return False

# 3D convolution layers
class Layer3D:

  #
  def __init__(self, input, filter, padding, stride, dilation, id=-1, network="unknown"):

    self.params = OrderedDict([
      ('n', input[0]),
      ('d', input[1]),
      ('h', input[2]),      
      ('w', input[3]),
      ('c', input[4]),

      ('k', filter[0]),
      ('t', filter[1]),
      ('r', filter[2]),
      ('s', filter[3]),

      ('pad_d', padding[0]),
      ('pad_h', padding[1]),
      ('pad_w', padding[2]),

      ('stride_d', stride[0]),
      ('stride_h', stride[1]),
      ('stride_w', stride[2]),

      ('dilation_d', dilation[0]),
      ('dilation_h', dilation[1]),
      ('dilation_w', dilation[2]),
    ])

    self.id = id
    self.network = network

  # returns profiler command for the layer
  def profiler_cmd(self, batch_size=1):
    
    cmdline_layer = " ".join(str("--%s=%d" % (key, value)) for (key, value) in self.params.items())

    if batch_size > 1:
      cmdline_layer = re.sub(r'--n=\d+', "".join(['--n=', str(batch_size)]), cmdline_layer.rstrip())

    return cmdline_layer

  def __str__(self):
    layer_details = \
      " %s" % (self.network) +\
      " layer id: %d," % (self.id)+\
      " input: (n=%d, d=%d, h=%d,w=%d,c=%d)," % (self.params['n'], self.params['d'], self.params['h'], self.params['w'], self.params['c']) +\
      " filter: (k=%d, t=%d, r=%d, s=%d, c=%d)," % (self.params['k'], self.params['t'], self.params['r'], self.params['s'], self.params['c']) +\
      " padding: (d=%d, h=%d, w=%d)," % (self.params['pad_d'], self.params['pad_h'], self.params['pad_w']) +\
      " stride: (d=%d, h=%d,w=%d)," % (self.params['stride_d'], self.params['stride_h'], self.params['stride_w']) +\
      " dilation: (d=%d, h=%d,w=%d)," % (self.params['dilation_d'], self.params['dilation_h'], self.params['dilation_w'])
    return layer_details

  # returns True if the layer is strided layer
  def is_strided(self):
    if self.params['stride_h'] != 1 or self.params['stride_w'] != 1 or self.params['stride_d'] != 1:
      return True
    return False

############################################################
# Initialize standard conv2d network and add layers
############################################################

############################################################
# Add Resnet50 layers
############################################################
Resnet50 = Network("Resnet50")

'''
TODO: Write/reuse parser from the old script layer file
"ResNet50_v1_conv0"   =  dimA:"1,3,224,224" * filtA:"64,3,7,7" * pad_h:3 * pad_w:3 * u:2 * v:2 
"ResNet50_v1_conv1"   =  dimA:"1,64,56,56" * filtA:"256,64,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv2"   =  dimA:"1,64,56,56" * filtA:"64,64,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv3"   =  dimA:"1,64,56,56" * filtA:"64,64,3,3" * pad_h:1 * pad_w:1 * u:1 * v:1 
"ResNet50_v1_conv4"   =  dimA:"1,256,56,56" * filtA:"64,256,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv5"   =  dimA:"1,256,56,56" * filtA:"512,256,1,1" * pad_h:0 * pad_w:0 * u:2 * v:2 
"ResNet50_v1_conv6"   =  dimA:"1,256,56,56" * filtA:"128,256,1,1" * pad_h:0 * pad_w:0 * u:2 * v:2 
"ResNet50_v1_conv7"   =  dimA:"1,128,28,28" * filtA:"128,128,3,3" * pad_h:1 * pad_w:1 * u:1 * v:1 
"ResNet50_v1_conv8"   =  dimA:"1,128,28,28" * filtA:"512,128,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv9"   =  dimA:"1,512,28,28" * filtA:"128,512,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv10"  =  dimA:"1,512,28,28" * filtA:"1024,512,1,1" * pad_h:0 * pad_w:0 * u:2 * v:2 
"ResNet50_v1_conv11"  =  dimA:"1,512,28,28" * filtA:"256,512,1,1" * pad_h:0 * pad_w:0 * u:2 * v:2 
"ResNet50_v1_conv12"  =  dimA:"1,256,14,14" * filtA:"256,256,3,3" * pad_h:1 * pad_w:1 * u:1 * v:1 
"ResNet50_v1_conv13"  =  dimA:"1,256,14,14" * filtA:"1024,256,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv14"  =  dimA:"1,1024,14,14" * filtA:"256,1024,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv15"  =  dimA:"1,1024,14,14" * filtA:"2048,1024,1,1" * pad_h:0 * pad_w:0 * u:2 * v:2 
"ResNet50_v1_conv16"  =  dimA:"1,1024,14,14" * filtA:"512,1024,1,1" * pad_h:0 * pad_w:0 * u:2 * v:2 
"ResNet50_v1_conv17"  =  dimA:"1,512,7,7" * filtA:"512,512,3,3" * pad_h:1 * pad_w:1 * u:1 * v:1 
"ResNet50_v1_conv18"  =  dimA:"1,512,7,7" * filtA:"2048,512,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
"ResNet50_v1_conv19"  =  dimA:"1,2048,7,7" * filtA:"512,2048,1,1" * pad_h:0 * pad_w:0 * u:1 * v:1 
'''

'''
# _Resnet50 first layer (layer_id = 0) with channel = 3 is not supported in cutlass
_Resnet50.add_layer(Layer2D(      
  [1, 224, 224, 3],  \
  [64, 7, 7, 3], \
  [3, 3, 3, 3], \
  [2, 2], \
  [1, 1], \
  0           # optional layer_id (all subsequent layer_id will have previous.layer.id + 1)
));
'''

Resnet50.add_layer(Layer2D(   
  [1, 56, 56, 64], \
  [256, 1, 1, 64], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1], \
  1           # optional layer_id (all subsequent layer_id will have previous.layer.id + 1)
));

Resnet50.add_layer(Layer2D(   
  [1, 56, 56, 64], \
  [64, 1, 1, 64], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 56, 56, 64], \
  [64, 3, 3, 64], \
  [1, 1, 1, 1], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 56, 56, 256], \
  [64, 1, 1, 256], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 56, 56, 256], \
  [512, 1, 1, 256], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 56, 56, 256], \
  [128, 1, 1, 256], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 28, 28, 128], \
  [128, 3, 3, 128], \
  [1, 1, 1, 1], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 28, 28, 128], \
  [512, 1, 1, 128], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 28, 28, 512], \
  [128, 1, 1, 512], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));
 
Resnet50.add_layer(Layer2D(   
  [1, 28, 28, 512], \
  [1024, 1, 1, 512], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));
        
Resnet50.add_layer(Layer2D(   
  [1, 28, 28, 512], \
  [256, 1, 1, 512], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 14, 14, 256], \
  [256, 3, 3, 256], \
  [1, 1, 1, 1], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 14, 14, 256], \
  [1024, 1, 1, 256], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 14, 14, 1024], \
  [256, 1, 1, 1024], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 14, 14, 1024], \
  [2048, 1, 1, 1024], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
))

Resnet50.add_layer(Layer2D(   
  [1, 14, 14, 1024], \
  [512, 1, 1, 1024], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 7, 7, 512], \
  [512, 3, 3, 512], \
  [1, 1, 1, 1], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 7, 7, 512], \
  [2048, 1, 1, 512], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

Resnet50.add_layer(Layer2D(   
  [1, 7, 7, 2048], \
  [512, 1, 1, 2048], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));


#########################################################################
# Strided layers only layers with stride = {2,2}
#########################################################################


StridedLayers = Network("StridedLayers")

#
# ResNet50 strided layers - (6 layers with 1x1 filter and 2x2 stride)
#

# Resnet50 layer 5
StridedLayers.add_layer(Layer2D(   
  [1, 56, 56, 256], \
  [512, 1, 1, 256], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1],
  1
));

# Resnet50 layer 6
StridedLayers.add_layer(Layer2D(   
  [1, 56, 56, 256], \
  [128, 1, 1, 256], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# Resnet50 layer 10
StridedLayers.add_layer(Layer2D(   
  [1, 28, 28, 512], \
  [1024, 1, 1, 512], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# Resnet50 layer 11    
StridedLayers.add_layer(Layer2D(   
  [1, 28, 28, 512], \
  [256, 1, 1, 512], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# Resnet50 layer 15
StridedLayers.add_layer(Layer2D(   
  [1, 14, 14, 1024], \
  [2048, 1, 1, 1024], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
))

# Resnet50 layer 16
StridedLayers.add_layer(Layer2D(   
  [1, 14, 14, 1024], \
  [512, 1, 1, 1024], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));


#
# ResNet50 v2 strided layers - (3 layers with 3x3 filter and 2x2 stride)
#
'''
"rn50_resnet_dgrad_006_R1" = dgrad_base * x: * dimA:"1,128,56,56" * filtA:"128,128,3,3" * padA:"1,1" * convStrideA:"2,2" * dilationA:"1,1"
"rn50_resnet_dgrad_012_R1" = dgrad_base * x: * dimA:"1,256,28,28" * filtA:"256,256,3,3" * padA:"1,1" * convStrideA:"2,2" * dilationA:"1,1"
"rn50_resnet_dgrad_018_R1" = dgrad_base * x: * dimA:"1,512,14,14" * filtA:"512,512,3,3" * padA:"1,1" * convStrideA:"2,2" * dilationA:"1,1"
'''

# rn50_resnet_dgrad_006_R1
StridedLayers.add_layer(Layer2D(   
  [1, 56, 56, 128], \
  [128, 3, 3, 128], \
  [1, 1, 1, 1], \
  [2, 2], \
  [1, 1]
));

# rn50_resnet_dgrad_012_R1
StridedLayers.add_layer(Layer2D(   
  [1, 28, 28, 256], \
  [256, 3, 3, 256], \
  [1, 1, 1, 1], \
  [2, 2], \
  [1, 1]
));

# rn50_resnet_dgrad_018_R1
StridedLayers.add_layer(Layer2D(   
  [1, 14, 14, 512], \
  [512, 3, 3, 512], \
  [1, 1, 1, 1], \
  [2, 2], \
  [1, 1]
));

#
# MaskRCNN strided layers - (6 layers with 1x1 filter and 2x2 stride)
#

# MaskRCNN strided layer
StridedLayers.add_layer(Layer2D(   
  [1, 336, 200, 256], \
  [128, 1, 1, 256], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# MaskRCNN strided layer
StridedLayers.add_layer(Layer2D(   
  [1, 336, 200, 256], \
  [512, 1, 1, 256], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# MaskRCNN strided layer
StridedLayers.add_layer(Layer2D(   
  [1, 168, 100, 512], \
  [256, 1, 1, 512], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# MaskRCNN strided layer
StridedLayers.add_layer(Layer2D(   
  [1, 168, 100, 512], \
  [1024, 1, 1, 512], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# MaskRCNN strided layer
StridedLayers.add_layer(Layer2D(   
  [1, 84, 50, 1024], \
  [512, 1, 1, 1024], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));

# MaskRCNN strided layer
StridedLayers.add_layer(Layer2D(   
  [1, 84, 50, 1024], \
  [2048, 1, 1, 1024], \
  [0, 0, 0, 0], \
  [2, 2], \
  [1, 1]
));


#########################################################################
# Large Strided layers only layers with stride > {2,2}
# Typically, stride = {4,4}, {8,8}, {32,32}
#########################################################################


LargeStridedLayers = Network("LargeStridedLayers")

# -Rdgrad -Pinh -Pouth -Pcomps -x -dimA18,32,120,160 -filtA48,32,4,4 -padA0,0 -convStrideA4,4 -dilationA1,1 -A1 -B0 -b -d0 -T-1
LargeStridedLayers.add_layer(Layer2D(   
  [18, 120, 160, 32], \
  [48, 4, 4, 32], \
  [0, 0, 0, 0], \
  [4, 4], \
  [1, 1],
  1
));


# -Rdgrad -A1 -B0 -b -Pinh -Pcomps -Pouth -formatIn0 -filtFormat0 -formatOut0 -dimA1,21,216,216 -filtA21,21,16,16 -padA0,0 -convStrideA8,8 -d0 -T-1
# C,K=21, padded to C,K=24 for satisfy alignment constraints for F16 C%8 and K%8
LargeStridedLayers.add_layer(Layer2D(   
  [8, 216, 216, 24], \
  [24, 16, 16, 24], \
  [0, 0, 0, 0], \
  [8, 8], \
  [1, 1]
));


# -Rdgrad -A1 -B0 -b -Pinh -Pcomps -Pouth -formatIn0 -filtFormat0 -formatOut0 -dimA1,32,128,128 -filtA32,1,16,16 -groupCount32 -u8 -v8 -padA0,0 -n64 -d0 -T-1
# ignorning groupCount
LargeStridedLayers.add_layer(Layer2D(   
  [64, 128, 128, 32], \
  [32, 16, 16, 32], \
  [0, 0, 0, 0], \
  [8, 8], \
  [1, 1]
));


# -Rdgrad -A1 -B0 -b -Pins -Pcomps -Pouts -formatIn1 -filtFormat1 -formatOut1 -x -dimA1,8,128,128 -filtA8,1,64,64 -groupCount8 -u32 -v32 -padA0,0 -n32 -d0 -T-1
# ignorning groupCount
LargeStridedLayers.add_layer(Layer2D(   
  [32, 128, 128, 8], \
  [8, 64, 64, 8], \
  [0, 0, 0, 0], \
  [32, 32], \
  [1, 1]
));


#########################################################################
# Add VGG-16 layers
#########################################################################
VGG_16 = Network("VGG_16")

VGG_16.add_layer(Layer2D(   
  [1, 224, 224, 64],
  [64, 3, 3, 64],
  [1, 1, 1, 1],
  [1, 1],
  [1, 1]
));

VGG_16.add_layer(Layer2D(   
  [1, 112, 112, 64],         # N, H, W, C
  [128, 3, 3, 64],           # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

VGG_16.add_layer(Layer2D(   
  [1, 112, 112, 128],        # N, H, W, C
  [128, 3, 3, 128],          # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

VGG_16.add_layer(Layer2D(   
  [1, 56, 56, 128],          # N, H, W, C
  [256, 3, 3, 128],          # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

VGG_16.add_layer(Layer2D(   
  [1, 56, 56, 256],          # N, H, W, C
  [256, 3, 3, 256],          # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

VGG_16.add_layer(Layer2D(   
  [1, 28, 28, 256],          # N, H, W, C
  [512, 3, 3, 256],          # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

VGG_16.add_layer(Layer2D(   
  [1, 28, 28, 512],          # N, H, W, C
  [512, 3, 3, 512],          # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

VGG_16.add_layer(Layer2D(   
  [1, 14, 14, 512],          # N, H, W, C
  [512, 3, 3, 512],          # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

VGG_16.add_layer(Layer2D(   
  [1, 7, 7, 512],            # N, H, W, C
  [512, 3, 3, 512],          # K, R, S, C
  [1, 1, 1, 1],              # pad
  [1, 1],                    # stride
  [1, 1]                     # dilation
));

############################################################
# Add VNet layers
VNet = Network("VNet")


VNet.add_layer(Layer3D(
  [1, 32, 32, 32, 16], \
  [64, 2, 2, 2, 16], \
  [0, 0, 0], \
  [2, 2, 2], \
  [1, 1, 1], \
  1
));

VNet.add_layer(Layer3D(
  [1, 16, 16, 16, 64], \
  [64, 3, 3, 3, 64], \
  [1, 1, 1], \
  [1, 1, 1], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 16, 16, 16, 32], \
  [128, 2, 2, 2, 32], \
  [0, 0, 0], \
  [2, 2, 2], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 8, 8, 8, 128], \
  [128, 3, 3, 3, 128], \
  [1, 1, 1], \
  [1, 1, 1], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 8, 8, 8, 64], \
  [128, 2, 2, 2, 64], \
  [0, 0, 0], \
  [2, 2, 2], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 4, 4, 4, 128], \
  [128, 3, 3, 3, 128], \
  [1, 1, 1], \
  [1, 1, 1], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 8, 8, 8, 64], \
  [64, 3, 3, 3, 64], \
  [1, 1, 1], \
  [1, 1, 1], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 16, 16, 16, 32], \
  [64, 2, 2, 2, 32], \
  [0, 0, 0], \
  [2, 2, 2], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 16, 16, 16, 32], \
  [32, 3, 3, 3, 32], \
  [1, 1, 1], \
  [1, 1, 1], \
  [1, 1, 1], \
));

VNet.add_layer(Layer3D(
  [1, 32, 32, 32, 16], \
  [32, 2, 2, 2, 16], \
  [0, 0, 0], \
  [2, 2, 2], \
  [1, 1, 1], \
));

#########################################################################
# Add Darpa RFMLS network
#########################################################################
DarpaNet = Network("DarpaNet")

DarpaNet.add_layer(Layer2D(
  [16, 1024, 256, 16], \
  [32, 1, 1, 16], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1], \
  1           # optional layer_id (all subsequent layer_id will have previous.layer.id + 1)
));

DarpaNet.add_layer(Layer2D(
  [16, 512, 128, 32], \
  [64, 1, 1, 32], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

DarpaNet.add_layer(Layer2D(
  [16, 256, 64, 64], \
  [64, 3, 3, 64], \
  [1, 1, 1, 1], \
  [1, 1], \
  [1, 1]
));

DarpaNet.add_layer(Layer2D(
  [16, 256, 64, 64], \
  [64, 3, 3, 64], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));

DarpaNet.add_layer(Layer2D(
  [16, 1024, 256, 64], \
  [64, 3, 3, 64], \
  [0, 0, 0, 0], \
  [1, 1], \
  [1, 1]
));
