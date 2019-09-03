# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import time
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n



class ClassificationMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val_prefix=''):
        self.val_prefix = val_prefix
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.mcc = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self, tp, tn, fp, fn):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        self.acc = (self.tp + self.tn) / ((self.tp + self.tn + self.fp + self.fn) or 1.0)
        self.mcc = (self.tp * self.tn - self.fp * self.fn) / (math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)) or 1.0)
        self.precision = self.tp / ((self.tp + self.fp) or 1.0)
        self.recall = self.tp / ((self.tp + self.fn) or 1.0)
        self.f1 = 2 * self.precision * self.recall / ((self.precision + self.recall) or 1.0)

    def vals(self):
        def attach_prefix(s):
            return '{}_{}'.format(self.val_prefix, s) if len(self.val_prefix) > 0 else s
        return [
            (attach_prefix('tp'), self.tp),
            (attach_prefix('tn'), self.tn),
            (attach_prefix('fp'), self.fp),
            (attach_prefix('fn'), self.fn),
            (attach_prefix('acc'), self.acc),
            (attach_prefix('mcc'), self.mcc),
            (attach_prefix('f1'), self.f1),
        ]


class F1Meter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val_prefix=''):
        self.val_prefix = val_prefix
        self.reset()

    def reset(self):
        self.n_pred = 0
        self.n_gold = 0
        self.n_correct = 0

        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self, n_pred, n_gold, n_correct):
        self.n_pred += n_pred
        self.n_gold += n_gold
        self.n_correct += n_correct

        if self.n_correct == 0:
            self.precision = 0
            self.recall = 0
            self.f1 = 0
        else:
            self.precision = self.n_correct * 1.0 / self.n_pred
            self.recall = self.n_correct * 1.0 / self.n_gold
            if self.precision + self.recall > 0:
                self.f1 = 2.0 * self.precision * self.recall / (self.precision + self.recall)
            else:
                self.f1 = 0.0

    def vals(self):
        def attach_prefix(s):
            return '{}_{}'.format(self.val_prefix, s) if len(self.val_prefix) > 0 else s
        return [
            (attach_prefix('npred'), self.n_pred),
            (attach_prefix('ngold'), self.n_gold),
            (attach_prefix('ncorr'), self.n_correct),
            (attach_prefix('p'), self.precision),
            (attach_prefix('r'), self.recall),
            (attach_prefix('f1'), self.f1),
        ]


class TypingMeter(object):
    """entity typing meters"""

    def __init__(self, val_prefix=''):
        self.val_prefix = val_prefix
        self.reset()

    def f1(self, p, r):
        if p == 0 or r == 0:
            return 0.0
        else:
            return 2.0 * p * r / (p + r)

    def reset(self):
        self.sample_num = 0
        
        # strict acc
        self.strict_acc_sum = 0
        self.strict_acc = 0

        # macro metrics
        self.ma_p_sum = 0
        self.ma_r_sum = 0
        self.ma_p = 0
        self.ma_r = 0
        self.ma_f1 = 0

        # micro metrics
        self.n_pred = 0
        self.n_true = 0
        self.n_corr = 0
        self.mi_p = 0
        self.mi_r = 0
        self.mi_f1 = 0

    def update(self, n_pred, n_true, n_corr, ma_p, ma_r, strict_acc, n=1):
        self.n_pred += n_pred
        self.n_true += n_true
        self.n_corr += n_corr

        self.ma_p_sum += ma_p
        self.ma_r_sum += ma_r

        self.sample_num += n
        self.strict_acc_sum += strict_acc

        self.strict_acc = self.strict_acc_sum / self.sample_num

        if self.n_pred > 0:
            self.mi_p = self.n_corr / self.n_pred
        else:
            self.mi_p = 0
        self.mi_r = self.n_corr / self.n_true
        self.mi_f1 = self.f1(self.mi_p, self.mi_r)

        self.ma_p = self.ma_p_sum / self.sample_num
        self.ma_r = self.ma_r_sum / self.sample_num
        self.ma_f1 = self.f1(self.ma_p, self.ma_r)

    def vals(self):
        def attach_prefix(s):
            return '{}_{}'.format(self.val_prefix, s) if len(self.val_prefix) > 0 else s
        return [
            (attach_prefix('acc'), self.strict_acc),
            (attach_prefix('ma_p'), self.ma_p),
            (attach_prefix('ma_r'), self.ma_r),
            (attach_prefix('ma_f1'), self.ma_f1),
            (attach_prefix('mi_p'), self.mi_p),
            (attach_prefix('mi_r'), self.mi_r),
            (attach_prefix('mi_f1'), self.mi_f1),
            (attach_prefix('valid_num'), self.sample_num),
        ]
