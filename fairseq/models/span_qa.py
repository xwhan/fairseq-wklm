import torch.nn as nn

from fairseq.tasks.masked_lm import MaskedLMTask
from fairseq.tasks.kdn_task import KDNTask
from . import (
    BaseFairseqModel, register_model, register_model_architecture,
)

from fairseq import checkpoint_utils


@register_model('span_qa')
class SpanQA(BaseFairseqModel):
    def __init__(self, args, pretrain_model):
        super().__init__()

        self.pretrain_model = pretrain_model
        self.qa_outputs = nn.Linear(args.model_dim, 2)
        self.use_kdn = args.use_kdn

        self.reset_parameters()

    def reset_parameters(self):
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()

    def forward(self, text, segment, paragraph_mask):
        if self.use_kdn:
            x = self.pretrain_model(text, segment, only_states=self.use_kdn)
        else:
            x, _ = self.pretrain_model(text, segment)

        logits = self.qa_outputs(x)
        if paragraph_mask.size(1) > x.size(1):
            paragraph_mask = paragraph_mask[:, :x.size(1)]
        assert [paragraph_mask[i].any() for i in range(paragraph_mask.size(0))]
        start, end = logits.split(1, dim=-1)
        return start, end, paragraph_mask

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bert-path', metavar='PATH', help='path to pretrained bert model')
        parser.add_argument('--model-dim', type=int, metavar='N', help='decoder input dimension')
        parser.add_argument('--last-dropout', type=float, metavar='D', help='dropout before projection')
        parser.add_argument('--use-kdn', action="store_true")
        # parser.add_argument('--model-dropout', type=float, metavar='D', help='lm dropout')
        # parser.add_argument('--attention-dropout', type=float, metavar='D', help='lm dropout')
        # parser.add_argument('--relu-dropout', type=float, metavar='D', help='lm dropout')
        # parser.add_argument('--proj-unk', action='store_true', help='if true, also includes unk emb in projection')
        # parser.add_argument('--layer-norm', action='store_true', help='if true, does non affine layer norm before proj')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        dictionary = task.dictionary

        assert args.bert_path is not None
        args.short_seq_prob = 0.0

        # # load from pretrained bert
        # task = MaskedLMTask(args, dictionary)
        # models, _ = checkpoint_utils.load_model_ensemble(
        # [args.bert_path], arg_overrides={
        #     'remove_head': True, 'share_encoder_input_output_embed': False
        # }, task=task)

        # load from pretrained kdn model
        task = KDNTask(args, dictionary)
        models, _ = checkpoint_utils.load_model_ensemble(
        [args.bert_path], task=task)


        assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'
        model = models[0]
        return SpanQA(args, model)


@register_model_architecture('span_qa', 'span_qa')
def base_architecture(args):
    args.model_dim = getattr(args, 'model_dim', 768)
