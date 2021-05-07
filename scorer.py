import sys
import torch
import torch.nn as nn
from collections import defaultdict
from math import log
import numpy as np

class Scorer(object):
    '''def __init__(self, char_list, model_path, rnn_type, ninp, nhid, nlayers, device):
        char_list = list(char_list) + ['sil_start', 'sil_end']
        self.inv_vocab_map = dict([(i, c) for (i, c) in enumerate(char_list)])
        self.vocab_map = dict([(c, i) for (i, c) in enumerate(char_list)])
        self.device = device
        self.history = defaultdict(tuple)'''

    def __init__(self, languageModel, tokenizer, device):
        self.languageModel = languageModel
        self.tokenizer = tokenizer
        self.device = device
        self.languageModel.to(self.device)

    def get_score(self, string):
        tokenize_input = self.tokenizer.tokenize(string)
        
        if len(tokenize_input) < 1:
            tokenize_input.append('<unk>')

        #print ("tokenize_input: " + str(tokenize_input))
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)]).to(self.device)
        #print ("tensor_input: " + str(tensor_input))
        '''outputs=self.languageModel(tensor_input, labels=tensor_input)

        print ("outputs.loss: " + str(outputs.loss))
        return -log(abs(outputs.loss) + 1e-6), outputs.loss'''
        outputs = self.languageModel(tensor_input, labels=tensor_input)
        loss = outputs.loss
        
        return -(len(tokenize_input) - 1) * loss.item(), loss


    def get_score_fast(self, strings):
        strings = [''.join(x) for x in strings]
        scores = [self.get_score(string)[0] for string in strings]
        return scores

