import sys
import torch
import torch.nn as nn
from collections import defaultdict
import math
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
        self.languageModel.eval()
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
        if math.isnan(loss.item()) == True:
            loss = 0
        else:
            loss = loss.item()
        return -(len(tokenize_input) - 1) * loss, loss


    def get_score_fast(self, strings):
        '''strings = [''.join(x) for x in strings]
        scores = [self.get_score(string)[0] for string in strings]
        return scores'''
        strings = [''.join(x) for x in strings]
        history_to_update = defaultdict(lambda: 0.0)
        scores = []
        for string in strings:
            print (string)
            if len(string) <= 2:
                score, hidden_state = self.get_score(string)
                scores.append(score)
                history_to_update[string] = score
            elif string in self.history:
                history_to_update[string] = self.history[string]
                scores.append(self.history[string][0])
            elif string[:-1] in self.history:
                score = self.history[string[:-1]]
                loss = self.get_score(string)[1]
                history_to_update[string] = score-loss
                scores.append(score-loss)
            else:
                raise ValueError("%s not stored" % (string[:-1]))
        self.history = history_to_update
        return scores


