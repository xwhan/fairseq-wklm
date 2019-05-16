# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data import Dictionary


class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
    ):
        super().__init__(pad, eos, unk)
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index

class BertDictionary(Dictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """
    def __init__(
        self,
        pad='[PAD]',
        unk='[UNK]',
        cls='[CLS]',
        mask='[MASK]',
        sep='[SEP]'
    ):
        super().__init__(pad, unk)
        (
            self.cls_word,
            self.mask_word,
            self.sep_word
        ) = cls, mask, sep
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of cls symbol"""
        idx = self.add_symbol(self.cls_word)
        return idx

    def sep(self):
        """Helper to get index of sep symbol"""
        idx = self.add_symbol(self.sep_word)
        return idx

    def mask(self): 
        """Helper to get index of sep symbol"""
        idx = self.add_symbol(self.mask_word)
        return idx
