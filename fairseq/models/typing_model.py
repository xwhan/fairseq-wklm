import torch.nn as nn
import torch

from fairseq.tasks.masked_lm import MaskedLMTask
from fairseq.tasks.kdn_task import KDNTask
from . import (
    BaseFairseqModel, register_model, register_model_architecture,
)

# from fairseq.models.hf_bert import PreTrainedBertModel

from pytorch_transformers import BertModel

from fairseq import checkpoint_utils


@register_model('typing')
class Typing(BaseFairseqModel):
    def __init__(self, args, pretrain_model):
        super().__init__()

        self.pretrain_model = pretrain_model
        self.use_kdn = args.use_kdn

        self.typing_outputs = nn.Linear(args.model_dim, args.num_class)
        self.last_dropout = nn.Dropout(args.last_drop)

        self.reset_parameters()

    def reset_parameters(self):
        self.typing_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.typing_outputs.bias.data.zero_()

    def forward(self, sentence, segment, entity_masks=None):
        """
        entity_masks: B, L
        outputs: B, num_class
        """

        if self.use_kdn:
            x = self.pretrain_model(
                sentence, segment, cls_rep=self.use_kdn)
        else:
            _, outputs = self.pretrain_model(sentence, segment)
            x = outputs["pooled_output"]

        sent_rep = self.last_dropout(x)
        typing_logits = self.typing_outputs(sent_rep)

        return typing_logits

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bert-path', metavar='PATH',
                            help='path to elmo model')
        parser.add_argument('--model-dim', type=int,
                            metavar='N', help='decoder input dimension')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        dictionary = task.dictionary

        # load from pretrained kdn model
        if args.use_kdn:
            print(f'| fine-tuning kdn pretrained model...')
            task = KDNTask(args, dictionary)
            models, _ = checkpoint_utils.load_model_ensemble(
                [args.bert_path], arg_overrides={"last_dropout": 0.0, "start_end": args.start_end, "boundary_loss": args.boundary_loss, "num_kdn": args.num_kdn, 'masking_ratio': args.masking_ratio}, task=task)
        else:
            print(f'| fine-tuning bert pretrained model...')
            task = MaskedLMTask(args, dictionary)
            models, _ = checkpoint_utils.load_model_ensemble(
                [args.bert_path], arg_overrides={
                    'remove_head': True, 'share_encoder_input_output_embed': False
                }, task=task)

        assert len(
            models) == 1, 'ensembles are currently not supported for elmo embeddings'
        model = models[0]
        return Typing(args, model)

@register_model_architecture('typing', 'typing')
def base_architecture(args):
    args.model_dim = getattr(args, 'model_dim', 768)
