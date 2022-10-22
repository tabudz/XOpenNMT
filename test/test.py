import torch
import torch.nn as nn
import onmt.model_builder as mb
import onmt
import json
import numpy as np 

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

opt_d = {'batch_size': 16,
 'gpu': -1,
 'model': 'model_step_1000.pt',
 'src': 'example.vocab.src',
 'src_onmttok_kwargs': '{"mode": "aggressive"}',
 'src_subword_model': 'subwords.en_de.bpe',
 'src_subword_type': 'bpe',
 'tgt': 'example.vocab.tgt',
 'tgt_onmttok_kwargs': '{"mode": "aggressive"}',
 'tgt_subword_model': 'subwords.en_de.bpe',
 'tgt_subword_type': 'bpe',
 'transforms': ['onmt_tokenize'],
 'fp32': False,
 'int8': False
}

opt = json.loads(json.dumps(opt_d), object_hook=obj)

checkpoint = torch.load('toy-ende/run/model_step_1000.pt', map_location=lambda storage, loc: storage)

fields, model, model_opt = mb.load_test_model(opt, 'toy-ende/run/model_step_1000.pt')
# print(model)
# print(model.encoder.embeddings.make_embedding(torch.LongTensor([[[1]]])))

embedding = model.encoder.embeddings.make_embedding(torch.LongTensor([[[1]]]))
# print(model.encoder.rnn(embedding))

def get_batch(source_l=3, bsize=1):
    # len x batch x nfeat
    test_src = torch.ones(source_l, bsize, 1).long()
    test_tgt = torch.ones(source_l, bsize, 1).long()
    test_length = torch.ones(bsize).fill_(source_l).long()
    return test_src, test_tgt, test_length

test_src, test_tgt, test_length = get_batch(source_l=3, bsize=1)
# print(test_src, test_tgt, test_length)
# print(model(test_src, test_tgt, test_length))

emb_s = model.encoder.embeddings.make_embedding(test_src)
# print(emb_s)

emb_s.retain_grad()
output, (hn, cn) = model.encoder.rnn(emb_s)

# print(cn)

emb_t = model.decoder.embeddings.make_embedding(None)
# print(emb_t)

decoder_input = torch.cat([emb_t.squeeze(0), output.squeeze(0)], 1)
print(decoder_input)

decoder_output, (hn, cn) = model.decoder.rnn(decoder_input, (hn, cn))
# print(decoder_output)
# print(hn)
# print(cn)

# decoder_output.sum().backward() # sum(): --> [y_pred1, y_pred2] --> F(y_pred1, y_pred2) --> sum
# Weighted Gradient
# Norm Gradient
# print(emb_s.grad)

#[y_pred1, y_pred2] --> F = sum(ypred1, ypred2) --> daoham F/daoham emb
# a1 = daoham y_pred1/ daoham emb1
# a2 = daoham y_pred2/ daoham emb2
# norm(a1, a2)
# weighted_sum(a1, a2)

# emb1 (c1 = attention * hidden? vs. emb input) + hidden --> RNN --> y_pred1
# emd2 

# print(torch.exp(model.generator(decoder_output)))

# print(model)

def forward(model, src, tgt, idx_tgt):
    emb_s = model.encoder.embeddings.make_embedding(src)
    emb_s.retain_grad()
    output, (hn, cn) = model.encoder.rnn(emb_s)
    emb_t = model.decoder.embeddings.make_embedding(tgt)
    decoder_input = torch.cat([emb_t.squeeze(0), hn.squeeze(0)], 1)
    decoder_output, (hn, cn) = model.decoder.rnn(decoder_input, (hn, cn))
    decoder_output[idx_tgt].backward()
    return emb_s

