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
        self.history = defaultdict(lambda: 0.0)

    def get_score(self, string):
        
        #print(string)
        
        tokenize_input = self.tokenizer.tokenize(string)
        #print ("tokenize_input: " + str(tokenize_input))

        
        while len(tokenize_input) < 2:
            tokenize_input.append('<pad>')


        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input[:-1])]).to(self.device)
        tensor_labels = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input[1:])]).to(self.device)
        
        outputs = self.languageModel(tensor_input, labels=tensor_labels)
        loss = outputs.loss
        if math.isnan(loss.item().cpu()) == True:
            loss = 100
        return -(len(tokenize_input) - 1) * loss.item(), loss


    def get_score_fast(self, strings):
        strings = [''.join(x) for x in strings]
        scores = [self.get_score(string)[0] for string in strings]
        return scores
