import sys
import torch
import torch.nn as nn
from collections import defaultdict


class Scorer(object):
    '''def __init__(self, char_list, model_path, rnn_type, ninp, nhid, nlayers, device):
        char_list = list(char_list) + ['sil_start', 'sil_end']
        self.inv_vocab_map = dict([(i, c) for (i, c) in enumerate(char_list)])
        self.vocab_map = dict([(c, i) for (i, c) in enumerate(char_list)])
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.rnn = RNN(rnn_type, len(char_list), ninp, nhid, nlayers).to(self.device)
        self.rnn.load_state_dict(torch.load(model_path))
        self.rnn.eval()
        self.history = defaultdict(tuple)'''

    def __init__(self, languageModel, tokenizer):
        self.languageModel = languageModel
        self.tokenizer = tokenizer

    def get_score(self, string):
        tokenize_input = self.tokenizer.tokenize(string)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=self.languageModel(tensor_input, lm_labels=tensor_input)
        return -len(string) * loss.item(), loss

    def get_score_fast(self, strings):
        strings = [''.join(x) for x in strings]
        scores = [self.get_score(string)[0] for string in strings]
        return scores

