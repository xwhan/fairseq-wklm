# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np

from . import BaseWrapperDataset


class ShuffleDataset(BaseWrapperDataset):

    def ordered_indices(self):
        return np.random.permutation(len(self))
