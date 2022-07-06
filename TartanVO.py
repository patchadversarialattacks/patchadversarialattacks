# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
import time

np.set_printoptions(precision=4, suppress=True, threshold=10000)

from Network.VONet import VONet

class TartanVO(object):
    def __init__(self, model_name, device):
        # import ipdb;ipdb.set_trace()
        self.vonet = VONet()
        self.device = device

        # load the whole model
        if model_name.endswith('.pkl'):
            modelname = 'models/' + model_name
            self.load_model(self.vonet, modelname)

        self.vonet.to(device=device)

        self.test_count = 0
        self.pose_std = torch.tensor([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013],
                                     dtype=torch.float32).to(device=device) # the output scale factor
        self.flow_norm = 20 # scale factor for flow

    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname, map_location=torch.device(self.device))
        model_dict = model.state_dict()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if ( 0 == len(preTrainDictTemp) ):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        print('Model loaded...')
        return model

    def test_batch(self, img1, img2, intrinsic, scale):
        self.test_count += 1

        # starttime = time.time()
        flow, pose = self.vonet(img1, img2, intrinsic)
        # inferencetime = time.time()-starttime
        pose_tensor = pose * self.pose_std # The output is normalized during training, now scale it back
        flow_tensor = flow * self.flow_norm


        # rescale to GT
        pose_tensor[:,:3] = scale.unsqueeze(1) * torch.nn.functional.normalize(pose_tensor.clone()[:,:3], p=2, dim=1)

        # print("{} Pose inference using {}s: \n{}".format(self.test_count, inferencetime, pose_tensor))
        # else:
        #     print('    scale is not given, using 1 as the default scale value..')

        return pose_tensor.squeeze(0), flow_tensor.squeeze(0)

    def __call__(self, img1, img2, intrinsic, scale):
        return self.test_batch(img1, img2, intrinsic, scale)

