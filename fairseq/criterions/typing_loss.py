import math
import torch.nn.functional as F
import torch.nn as nn

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('typing_loss')
class TypingCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.padding_idx = task.ignore_index
        self.max_length = task.max_length
        self.num_class = task.num_class
        self.loss_fn = nn.BCELoss()

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if sample['net_input']['sentence'].size(1) > self.max_length:
            assert False

        net_output = model(**sample['net_input'])
        loss, type_probs = self.compute_loss(model, net_output, sample)
        sample_size = sample['target'].size(0)
        logging_output = {
            'loss': utils.item(loss.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'probs': type_probs
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        typing_prob = F.sigmoid(net_output)

        loss = self.loss_fn(typing_prob, sample['target'].type(typing_prob.type()))
        return loss, typing_prob

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
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
