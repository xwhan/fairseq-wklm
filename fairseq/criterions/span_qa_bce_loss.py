# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from fairseq import utils
from torch.nn import BCELoss
import torch
import torch.nn.functional as F

from . import FairseqCriterion, register_criterion


@register_criterion('span_qa_bce')
class SpanQABCECriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True, is_valid=False):

        start_targets = sample['start_target']
        end_targets = sample['end_target']
        net_input = sample['net_input']
        start_out, end_out, paragraph_mask = model(**net_input)
        outs = (start_out, end_out)
        assert len(outs) == 2
        questions_mask = paragraph_mask.ne(1)

        paragraph_outs = [o.view(-1, o.size(1)).masked_fill(questions_mask,
                                                            torch.Tensor([-1e15]).half().item()) for o in outs]
        outs = [F.sigmoid(_) for _ in paragraph_outs]

        loss_fct = BCELoss(reduction='sum')
        start_loss = loss_fct(outs[0], start_targets.type(outs[0].type()))
        end_loss = loss_fct(outs[1], end_targets.type(outs[1].type()))
        npara_tokens = paragraph_mask.sum()
        loss = (start_loss + end_loss) / (2 * npara_tokens)

        # sample_size = sample['nsentences']
        sample_size = 1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        if is_valid:
            return loss, sample_size, logging_output, outs
        else:
            return loss, sample_size, logging_output

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
