import os
import shutil

import tokenizers
import pandas as pd

from types import SimpleNamespace
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

filepath = os.path.realpath(__file__)

class BERTweetTokenizer():
    
    def __init__(self,pretrained_path = 'pretrained_models/BERTweet_base_transformers/'):
        
        self.bpe = fastBPE(SimpleNamespace(bpe_codes= pretrained_path + "bpe.codes"))
        self.vocab = Dictionary()
        self.vocab.add_from_file(pretrained_path + "dict.txt")
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.pad_token = '<pad>'
        self.cls_token = '<s>'
        self.sep_token = '</s>'
        
    def bpe_encode(self,text):
        return self.bpe.encode(text)
    
    def encode(self,text,add_special_tokens=False):
        subwords = self.bpe.encode(text)
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    def tokenize(self,text):
        return self.bpe_encode(text).split()
    
    def convert_tokens_to_ids(self,tokens):
        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    def decode_tokens(self, tokens):
        decoded = ' '.join(tokens).replace('@@ ','').strip()
        return decoded

def custom_wp_tokenizer(corpus,text_filepath,tokenizer_save_path,vocab_size=10000,min_frequency=3):
    if type(corpus[0]) == list:
        corpus = [" ".join(i) for i in corpus]

    try:
        os.makedirs(text_filepath)
    except OSError:
        pass

    tokenizer = tokenizers.BertWordPieceTokenizer(
            vocab_file=None,
            unk_token='[UNK]',
            sep_token='[SEP]',
            cls_token='[CLS]',
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,
            lowercase=True,
            wordpieces_prefix='##'
        )#SentencePieceBPETokenizer()

    df = pd.DataFrame()
    df['text'] = corpus
    df.to_csv(os.path.join(text_filepath,'file.txt'),header=False,index=False)

    try:
        os.makedirs(tokenizer_save_path)
    except OSError:
        pass

    tokenizer.train(files=os.path.join(text_filepath,'file.txt'), vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]'])

    #tokenizer.save(directory=tokenizer_save_path,name='wpe')
    tokenizer.save_model(tokenizer_save_path, 'wpe')
    
    shutil.move(os.path.join(tokenizer_save_path,'wpe-vocab.txt'), os.path.join(tokenizer_save_path,'vocab.txt'))

    os.remove(os.path.join(text_filepath,'file.txt'))

def custom_bpe_tokenizer(corpus,text_filepath,tokenizer_save_path,vocab_size=10000,min_frequency=3):
    if type(corpus[0]) == list:
        corpus = [" ".join(i) for i in corpus]

    try:
        os.makedirs(text_filepath)
    except OSError:
        pass

    tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=None,
            merges_file=None,
        )#SentencePieceBPETokenizer()

    df = pd.DataFrame()
    df['text'] = corpus
    df.to_csv(os.path.join(text_filepath,'file.txt'),header=False,index=False)

    try:
        os.makedirs(tokenizer_save_path)
    except OSError:
        pass

    tokenizer.train(files=os.path.join(text_filepath,'file.txt'), vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]'])

    tokenizer.save(directory=tokenizer_save_path,name='bpe')

    shutil.move(os.path.join(tokenizer_save_path,'bpe-vocab.json'), os.path.join(tokenizer_save_path,'vocab.json'))
    shutil.move(os.path.join(tokenizer_save_path,'bpe-merges.txt'), os.path.join(tokenizer_save_path,'merges.txt'))

    os.remove(os.path.join(text_filepath,'file.txt'))