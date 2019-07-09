import itertools
import json
import os
import sys
import torch

from collections import defaultdict

sys.path.append('/private/home/namangoyal/src/gpt2_bpe')
sys.path.append('/private/home/namangoyal/fairseq-py')

from encoder import MultiprocessingEncoder
from fairseq.data.dictionary import Dictionary
from fairseq import checkpoint_utils, tasks, options
from fairseq.utils import move_to_cuda


class bpe_args:
    def __init__(self, encoder_json, vocab_bpe):
        self.encoder_json = encoder_json
        self.vocab_bpe = vocab_bpe
        self.min_len = None
        self.max_len = None
        self.keep_empty = True


def _get_input_ids(sentence, fairseq_dictionary):
    input_ids = []
    for token in sentence.split():
        input_ids.append(fairseq_dictionary.index(token))
    return input_ids


def _laod_models(args):
    task = tasks.setup_task(args)
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        task=task,
        arg_overrides={'remove_sentence_classification_head': False}
    )
    for model in models:
        model.eval()
    return models


def main(args):
    models = _laod_models(args)
    models[0].cuda()

    bpe_encoder = MultiprocessingEncoder(
        bpe_args(args.encoder_json, args.vocab_bpe)
    )
    bpe_encoder.initializer()

    candidate_samples = defaultdict(list)

    fairseq_dictionary = Dictionary.load(args.fairseq_dict)
    mask_idx = fairseq_dictionary.add_symbol('<mask>')

    for fname in ['train.jsonl', 'val.jsonl']:
        with open(os.path.join(args.datadir, fname)) as fin:
            for line in fin:
                sample = json.loads(line.strip())
                candidate_samples[
                    (sample['text'], sample['target']['span2_index'])
                ].append(
                    {
                        'span1_index': sample['target']['span1_index'],
                        'span1_text': sample['target']['span1_text'],
                        'label': sample['label'],
                        'idx': sample['idx'],
                        'pronoun': sample['target']['span2_text'],
                    }
                )
    candidate_samples = {
        k: v for k, v in candidate_samples.items() if len(v) > 1
    }
    filtered_candidate_samples = {}
    for text, candidates in candidate_samples.items():
        num_correct = sum([candidate['label'] for candidate in candidates])
        assert num_correct <= 1
        if num_correct == 1:
            filtered_candidate_samples[text] = candidates

    ncorrect = 0
    samples = 0

    for text, candidates in filtered_candidate_samples.items():
        for candidate in candidates:
            assert text[0].split(" ")[text[1]] == candidate['pronoun'].strip()

            before_pronoun = text[0].split(" ")[:text[1]]
            after_pronoun = text[0].split(" ")[text[1] + 1:]

            before_pronoun_string = " ".join(before_pronoun)
            after_pronoun_string = " ".join(after_pronoun)

            before_pronoun_bpe = bpe_encoder.encode_lines([before_pronoun_string])[1][0]
            after_pronoun_bpe = bpe_encoder.encode_lines([after_pronoun_string])[1][0]

            candidate_bpe = bpe_encoder.encode_lines([candidate['span1_text'].strip()])[1][0]

            before_pronoun_input_ids = _get_input_ids(before_pronoun_bpe, fairseq_dictionary)
            after_pronoun_input_ids = _get_input_ids(after_pronoun_bpe, fairseq_dictionary)
            candidate_input_ids = _get_input_ids(candidate_bpe, fairseq_dictionary)

            input_representation = list(
                itertools.chain(
                    [0],
                    before_pronoun_input_ids,
                    [mask_idx] * len(candidate_bpe.split(' ')),
                    after_pronoun_input_ids,
                    [fairseq_dictionary.eos()],
                )
            )

            sample = torch.tensor(input_representation).unsqueeze(0)
            sample = move_to_cuda(sample)

            mask_start_index = input_representation.index(mask_idx)
            mask_end_index = len(input_representation) - input_representation[::-1].index(mask_idx)

            with torch.no_grad():
                x, extra = models[0](src_tokens=sample)

                masked_lm_output = x.detach()[0, mask_start_index: mask_end_index, :].softmax(dim=-1)

            assert masked_lm_output.shape[0] == len(candidate_bpe.split(' '))

            masked_bpe_probs = []
            for i, masked_bpe_id in enumerate(candidate_input_ids):
                 masked_bpe_prob = masked_lm_output[i, masked_bpe_id]
                 masked_bpe_probs.append(masked_bpe_prob)

            candidate_prob = torch.tensor(masked_bpe_probs).mean()

            candidate['prob'] = candidate_prob

        for candidate in candidates:
            candidate['prediction'] = False
        sorted(candidates, key=lambda x:x['prob'], reverse=True)[0]['prediction'] = True
        # random.choice(candidates)['prediction'] = True

    for text, candidates in filtered_candidate_samples.items():
        for candidate in candidates:
            if candidate['label'] == candidate['prediction']:
                ncorrect += 1
            samples += 1

    print(f"| Accuracy: {(ncorrect * 1.0)/samples}, Num Samples: {samples}")


if __name__ == '__main__':
    parser = options.get_parser('Trainer', 'sentence_classification')

    options.add_dataset_args(parser)
    parser.add_argument('--criterion', default='sentence_classification')
    parser.add_argument('--path', metavar='FILE', help='path(s) to model file(s), colon separated')

    parser.add_argument('--datadir', default='/private/home/namangoyal/dataset/superglue/WSC/')
    parser.add_argument('--encoder-json', default='/private/home/namangoyal/src/gpt2_bpe/encoder.json')
    parser.add_argument('--vocab-bpe', default='/private/home/namangoyal/src/gpt2_bpe/vocab.bpe')
    parser.add_argument('--fairseq-dict', default='/private/home/myleott/data/data-bin/CC-NEWS-en.v5/dict.txt')

    args = options.parse_args_and_arch(parser)
    args = parser.parse_args()

    main(args)
