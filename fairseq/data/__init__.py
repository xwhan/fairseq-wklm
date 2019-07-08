# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from .dictionary import Dictionary, TruncatedDictionary
from .masked_lm_dictionary import BertDictionary, MaskedLMDictionary

from .fairseq_dataset import FairseqDataset

from .backtranslation_dataset import BacktranslationDataset
from .block_pair_dataset import BlockPairDataset
from .concat_dataset import ConcatDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .language_pair_dataset import LanguagePairDataset
from .lm_context_window_dataset import LMContextWindowDataset
from .masked_lm_dataset import MaskedLMDataset
from .monolingual_dataset import MonolingualDataset
from .noising import NoisingDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .transform_eos_lang_pair_dataset import TransformEosLangPairDataset
from .sentence_classification_dataset import SentenceClassificationDataset
from .sentence_pair_classification_dataset import SentencePairClassificationDataset
from .multi_choice_qa_dataset import MultiChoiceQADataset
from .squad_dataset import SquadDataset
from .paragraph_ranking_dataset import ParagraphRankingDataset
from .concat_sentences_dataset import ConcatSentencesDataset
from .id_dataset import IdDataset
from .nested_dictionary_dataset import NestedDictionaryDataset
from .num_samples_dataset import NumSamplesDataset
from .numel_dataset import NumelDataset
from .offset_tokens_dataset import OffsetTokensDataset
from .prepend_token_dataset import PrependTokenDataset
from .raw_label_dataset import RawLabelDataset
from .pad_dataset import RightPadDataset
from .sort_dataset import SortDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'BacktranslationDataset',
    'BertDictionary',
    'BlockPairDataset',
    'ConcatDataset',
    'CountingIterator',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'GroupedIterator',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'LMContextWindowDataset',
    'MaskedLMDataset',
    'MaskedLMDictionary',
    'MMapIndexedDataset',
    'MonolingualDataset',
    'NoisingDataset',
    'RoundRobinZipDatasets',
    'ShardedIterator',
    'TokenBlockDataset',
    'TransformEosDataset',
    'TransformEosLangPairDataset',
    'TruncatedDictionary',
    'SentencePairClassificationDataset',
    'SentenceClassificationDataset',
    'MultiChoiceQADataset',
    'IndexedInMemoryDataset',
    'SquadDataset',
    'ParagraphRankingDataset',
    'IdDataset',
    'NestedDictionaryDataset',
    'NumSamplesDataset',
    'OffsetTokensDataset',
    'NumelDataset',
    'PrependTokenDataset',
    'RawLabelDataset',
    'RightPadDataset',
    'SortDataset',
    'ConcatSentencesDataset'
]
