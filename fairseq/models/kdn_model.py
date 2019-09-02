import torch.nn as nn
import torch

from fairseq.tasks.masked_lm import MaskedLMTask
from . import (
    BaseFairseqModel, register_model, register_model_architecture,
)

from fairseq.modules import LayerNorm

from fairseq import checkpoint_utils, utils

@register_model('kdn')
class KDN(BaseFairseqModel):
    def __init__(self, args, pretrain_model):
        super().__init__()

        self.pretrain_model = pretrain_model
        self.padding_idx = pretrain_model.encoder.padding_idx

        self.add_layer = args.add_layer
        self.boundary = args.boundary_loss
        self.num_kdn_layer = args.num_kdn

        if self.add_layer:
            self.kdn_layers = self.pretrain_model.encoder.sentence_encoder.add_transformer_layer(
                self.num_kdn_layer)
            self.lm_head_transform_weight = nn.Linear(
                self.pretrain_model.encoder.sentence_encoder.embedding_dim, self.pretrain_model.encoder.sentence_encoder.embedding_dim)
            self.activation_fn = utils.get_activation_fn(
                self.pretrain_model.encoder.sentence_encoder.activation_fn)
            self.layer_norm = LayerNorm(
                self.pretrain_model.encoder.sentence_encoder.embedding_dim)

        self.start_end = args.start_end

        if self.start_end:
            self.kdn_s_outputs = nn.Linear(args.model_dim, 2) # aggregate CLS and entity tokens
            self.kdn_e_outputs = nn.Linear(args.model_dim, 2)
        if self.boundary:
            self.kdn_b_outputs = nn.Linear(args.model_dim*2, 2)
        if not self.boundary and not self.start_end:
            self.kdn_outputs = nn.Linear(args.model_dim, 2)

        self.kdn_drop = nn.Dropout(args.last_dropout)
        self.reset_parameters()

    def reset_parameters(self):

        if self.start_end:
            self.kdn_s_outputs.weight.data.normal_(mean=0.0, std=0.02)
            self.kdn_s_outputs.bias.data.zero_()
            self.kdn_e_outputs.weight.data.normal_(mean=0.0, std=0.02)
            self.kdn_e_outputs.bias.data.zero_()
        if self.boundary:
            self.kdn_b_outputs.weight.data.normal_(mean=0.0, std=0.02)
            self.kdn_b_outputs.bias.data.zero_()
        if not self.boundary and not self.start_end:
            self.kdn_outputs.weight.data.normal_(mean=0.0, std=0.02)
            self.kdn_outputs.bias.data.zero_()


    def forward(self, sentence, segment, entity_masks=None, only_states=False, cls_rep=False):
        """
        entity_masks: B, |E|, L
        outputs: B, |E|, 2
        """
        lm_logits, outputs = self.pretrain_model(sentence, segment)

        if self.add_layer:
            padding_masks = sentence.eq(self.padding_idx)
            last_states = outputs['inner_states'][-1]
            for layer in self.kdn_layers:
                last_states, _ = layer(last_states, self_attn_padding_mask=padding_masks)
            last_states = last_states.transpose(0, 1)
            x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(last_states)))
        else:
            x = outputs['task_specific']

        if only_states:
            if not self.add_layer:
                return x
            else:
                return torch.cat([x, outputs['task_specific']], dim=-1)
        elif cls_rep:
            return outputs['pooled_output']


        if self.start_end:
            start_masks = (entity_masks == 1).type(x.type())
            end_masks = (entity_masks == 2).type(x.type())
            start_tok_rep = torch.bmm(start_masks, x)
            end_tok_rep = torch.bmm(end_masks, x)
            start_logits = self.kdn_s_outputs(self.kdn_drop(start_tok_rep))
            end_logits = self.kdn_e_outputs(self.kdn_drop(end_tok_rep))
            return start_logits, end_logits, lm_logits
        
        if self.boundary:
            before_masks = (entity_masks == -1).type(x.type())
            after_masks = (entity_masks == -2).type(x.type())
            before_tok_rep = torch.bmm(before_masks, x)
            after_tok_rep = torch.bmm(after_masks, x)
            entity_rep = torch.cat([before_tok_rep, after_tok_rep], dim=-1)
            entity_logits = self.kdn_b_outputs(self.kdn_drop(entity_rep))
            return entity_logits, lm_logits

        if not self.boundary and not self.start_end:
            start_masks = (entity_masks == 1).type(x.type())
            start_tok_rep = torch.bmm(start_masks, x)
            entity_logits = self.kdn_outputs(self.kdn_drop(start_tok_rep))
            return entity_logits, lm_logits


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bert-path', metavar='PATH', help='path to elmo model')
        parser.add_argument('--model-dim', type=int, metavar='N', help='decoder input dimension')
        parser.add_argument('--last-dropout', type=float, metavar='D', default=0.0, help='dropout before projection')

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
