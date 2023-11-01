import os
import numpy as np
import pickle
import torch
from transformers import GPT2Tokenizer, GPT2Model

DIR = '/Users/tinam/LAMB/mp/analysis'
os.chdir(DIR)

model_name = 'gpt2-medium'   #gpt2, gpt2-medium, gpt2-large, gpt2-xl
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)
n_layer = model.config.n_layer
n_head = model.config.n_head

def tokenize_and_align(text):
    '''Given a text (as a string), returns a list of lists where each sub-list contains 
    tokenized tokens for the corresponding word.'''
    tokens = tokenizer.tokenize(text)
    i = 0
    words_to_tokens = [[]]
    for ind, t in enumerate(tokens):
        if t.startswith('Ġ'):
            words_to_tokens.append([ind])
            i += 1
        else:
            words_to_tokens[i].append(ind)
    assert len(text.split()) == len(words_to_tokens)

    return words_to_tokens

def get_word_word_attention(token_token_attention, words_to_tokens, mode='mean'):
    '''Convert token-token attention to word-word attention (when tokens are
    derived from words using something like byte-pair encodings).'''

    word_word_attention = token_token_attention
    not_word_starts = []
    for word in words_to_tokens:
        not_word_starts += word[1:]

    # sum up the attentions for all tokens in a word that has been split
    for word in words_to_tokens:
        word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

    # several options for combining attention maps for words that have been split
    for word in words_to_tokens:
        if mode == 'first':
            pass
        elif mode == 'mean':
            word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
        elif mode == 'max':
            word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
            word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
        else:
            raise ValueError('Unknown aggregation mode', mode)
    word_word_attention = np.delete(word_word_attention, not_word_starts, 0)

    return word_word_attention

with open('snts.txt', 'r') as f:
    snts = f.read().splitlines()

# get attention matrix at word level
attns = []
for s in snts:
    inputs = tokenizer(s, return_tensors='pt')
    outputs = model(**inputs, output_attentions=True).attentions
    attn_all = []
    for lyr in range(n_layer):
        attn = outputs[lyr].detach().squeeze().numpy()
        if attn.shape[1] != len(s.split()):
            words_to_tokens = tokenize_and_align(s)
            attn = np.array([get_word_word_attention(token_token_attention,words_to_tokens,mode='mean') 
                for token_token_attention in attn])
        attn_all.append(attn)
    attns.append(np.array(attn_all)) #snt*layer*head*num_word*num_word

pickle.dump(attns, open('gpt2_attns_med.p', 'wb'), protocol=2)