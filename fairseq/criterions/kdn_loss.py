# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('kdn_loss')
class KDN_loss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.ignore_index = task.ignore_index

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if sample['net_input']['sentence'].size(1) > 512:
            assert False

        entity_logits = model(**sample['net_input'])
        loss, lprobs = self.compute_loss(model, entity_logits, sample)
        n_entities = utils.strip_pad(sample['target'], self.ignore_index).numel()
        loss = loss / n_entities
        sample_size = 1
        logging_output = {
            'loss': utils.item(loss.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'n_entities': n_entities,
            'lprobs': lprobs
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, target, ignore_index=self.ignore_index, reduction="sum")
        return loss, lprobs

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
