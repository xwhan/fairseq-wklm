# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(-1), \
        "Logits and Targets tensor shapes don't match up"

    loss = F.nll_loss(
        F.log_softmax(logits, -1, dtype=torch.float32),
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss



@register_criterion('kdn_loss')
class KDN_loss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.ignore_index = task.ignore_index
        self.max_length = task.max_length
        self.use_mlm = task.use_mlm

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if sample['net_input']['sentence'].size(1) > self.max_length:
            assert False

        entity_logits, lm_logits = model(**sample['net_input'])
        if self.use_mlm:
            lm_logits = lm_logits.view(-1, lm_logits.size(-1))
            lm_targets = sample['lm_target'].view(-1)
            lm_loss = compute_cross_entropy_loss(lm_logits, lm_targets, self.padding_idx)
            ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()
            lm_loss = lm_loss / ntokens

        ent_loss, lprobs = self.compute_loss(model, entity_logits, sample)
        n_entities = utils.strip_pad(sample['target'], self.ignore_index).numel()
        ent_loss = ent_loss / n_entities
        loss = ent_loss + lm_loss if self.use_mlm else ent_loss

        sample_size = 1
        logging_output = {
            'loss': utils.item(loss.data),
            'ent_loss': utils.item(ent_loss.data),
            'lm_loss': utils.item(lm_loss.data),
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
        ent_loss_sum = sum(log.get('ent_loss', 0) for log in logging_outputs)
        lm_loss_sum = sum(log.get('lm_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ent_loss': ent_loss_sum / sample_size / math.log(2),
            'lm_loss': lm_loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
