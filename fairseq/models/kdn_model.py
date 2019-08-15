import torch.nn as nn
import torch

from fairseq.tasks.masked_lm import MaskedLMTask
from . import (
    BaseFairseqModel, register_model, register_model_architecture,
)

from fairseq import checkpoint_utils


@register_model('kdn')
class KDN(BaseFairseqModel):
    def __init__(self, args, pretrain_model):
        super().__init__()

        self.pretrain_model = pretrain_model

        if args.add_layer:
            self.kdn_layers = pretrain_model.sentence_encoderadd_transformer_layer(args.kdn_layers)

        self.start_end = args.start_end

        if self.start_end:
            self.kdn_s_outputs = nn.Linear(args.model_dim, 2) # aggregate CLS and entity tokens
            self.kdn_e_outputs = nn.Linear(args.model_dim, 2)
        else:
            self.kdn_outputs = nn.Linear(args.model_dim, 2)

        self.kdn_drop = nn.Dropout(args.last_dropout)

        self.reset_parameters()

    def reset_parameters(self):

        if self.start_end:
            self.kdn_s_outputs.weight.data.normal_(mean=0.0, std=0.02)
            self.kdn_s_outputs.bias.data.zero_()
            self.kdn_e_outputs.weight.data.normal_(mean=0.0, std=0.02)
            self.kdn_e_outputs.bias.data.zero_()
        else:
            self.kdn_outputs.weight.data.normal_(mean=0.0, std=0.02)
            self.kdn_outputs.bias.data.zero_()


    def forward(self, sentence, segment, entity_masks=None, only_states=False):
        """
        entity_masks: B, |E|, L
        outputs: B, |E|, 2
        """
        lm_logits, outputs = self.pretrain_model(sentence, segment)

        x = outputs['task_specific']

        if only_states:
            return x

        # # initial outputs, concatenate cls with average pool
        # cls_rep = x[:,0,:]
        # entity_masks = entity_masks.type(x.type())
        # entity_rep = torch.bmm(entity_masks, x)
        # entity_rep = torch.cat([cls_rep.unsqueeze(1).expand_as(entity_rep), entity_rep], dim=-1)
        # entity_logits = self.kdn_outputs(entity_rep)

        if self.start_end:
            start_masks = (entity_masks == 1).type(x.type())
            end_masks = (entity_masks == 2).type(x.type())
            start_tok_rep = torch.bmm(start_masks, x)
            end_tok_rep = torch.bmm(end_masks, x)
            start_logits = self.kdn_s_outputs(self.kdn_drop(start_tok_rep))
            end_logits = self.kdn_e_outputs(self.kdn_drop(end_tok_rep))
            return start_logits, end_logits, lm_logits
        else:
            start_masks = (entity_masks == 1).type(x.type())
            start_tok_rep = torch.bmm(start_masks, x)
            entity_logits = self.kdn_outputs(self.kdn_drop(start_tok_rep))
            return entity_logits, lm_logits


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bert-path', metavar='PATH', help='path to elmo model')
        parser.add_argument('--model-dim', type=int, metavar='N', help='decoder input dimension')
        parser.add_argument('--last-dropout', type=float, metavar='D', help='dropout before projection')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        dictionary = task.dictionary

        assert args.bert_path is not None
        task = MaskedLMTask(args, dictionary)

        models, _ = checkpoint_utils.load_model_ensemble([args.bert_path], task=task)
        assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'
        model = models[0]
        return KDN(args, model)


@register_model_architecture('kdn', 'kdn')
def base_architecture(args):
    args.model_dim = getattr(args, 'model_dim', 768)
