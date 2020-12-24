import os
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
import tensorflow as tf
import torch

def flatten(elems):
    return [e for elem in elems for e in elem]

def _get_unique(elems):
    if type(elems[0]) == list:
        corpus = flatten(elems)
    else:
        corpus = elems
    elems, freqs = zip(*Counter(corpus).most_common())
    return list(elems)

def convert_categorical_label_to_int(labels,save_path=None):
    if type(labels[0]) == list:
        uniq_labels = _get_unique(flatten(labels))
    else:
        uniq_labels = _get_unique(labels)

    if os.path.exists(save_path):
        label_to_id = pickle.load(open(save_path,'rb'))

    else:
        if type(labels[0]) == list:
            label_to_id = {w:i+1 for i,w in enumerate(uniq_labels)}
        else:
            label_to_id = {w:i for i,w in enumerate(uniq_labels)}

    new_labels = []
    if type(labels[0]) == list:
        for i in labels:
            new_labels.append([label_to_id[j] for j in i])
    else:
        new_labels = [label_to_id[j] for j in labels]

    if save_path:
        with open(save_path,'wb') as f:
            pickle.dump(label_to_id,f,-1)

    return new_labels, label_to_id

class CoNLLSeqData(object):
    def __init__(self, filepath, word_index=0, label_index=3):
        self.word_index = word_index
        self.label_index = label_index
        
        self.words, self.start_list, self.end_list  = self.read_conll_format(filepath)
        self.labels = self.read_conll_format_labels(filepath)

        #assert len(self.words) == len(self.labels)

        self.sentence = ["sentence_{}".format(i+1) for i in range(len(self.words))]
        

    def read_conll_format_labels(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                probs = line.split("\t")[self.label_index]
                post.append(probs)
                #print("post: ", post)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_conll_format(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        start_list, end_list, starts, ends = [], [], [], []
        start, end = 0, 0
        for line in lines:
            if line:
                start = end + 1
                words = line.split("\t")[self.word_index]
                end = start + len(words) 
                # print("words: ", words)
                post.append(words.lower())
                starts.append(start)
                ends.append(end)
            elif post:
                posts.append(post)
                start_list.append(starts)
                end_list.append(ends)
                post = []
                start, end = 0, 0
        # a list of lists of words/ labels
        return posts, start_list, end_list

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines
        
def _convert_to_transformer_inputs(text, tokenizer, max_sequence_length, text2=None, bertweettokenizer=False):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        if bertweettokenizer == False:
            try:
                inputs = tokenizer.encode_plus(str1, str2,
                    add_special_tokens=True,
                    max_length=length,
                    truncation_strategy=truncation_strategy, truncation=True)
            except:
                inputs = tokenizer.encode_plus(str1, str2,
                    add_special_tokens=True,
                    max_length=length,
                    truncation_strategy=truncation_strategy)

            input_ids =  inputs["input_ids"]
        else:
            input_ids = tokenizer.encode(str1)
            
            if len(input_ids) <= max_sequence_length-2:
                input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                if max_sequence_length - len(input_ids) > 0:
                    input_ids = input_ids + [tokenizer.pad_token_id]*(max_sequence_length - len(input_ids))
            else:
                input_ids = input_ids[:max_sequence_length-2]
                input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                if max_sequence_length - len(input_ids) > 0:
                    input_ids = input_ids + [tokenizer.pad_token_id]*(max_sequence_length - len(input_ids))
        
        input_masks = [1] * len(input_ids)
        input_segments = [0]*len(input_ids) #[0]*(len(input_ids)-1) + [1] #inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids, input_masks, input_segments = return_id(
        text, text2, 'longest_first', max_sequence_length)
    
    return [input_ids, input_masks, input_segments]

def compute_transformer_input_arrays(df, column1, tokenizer, max_sequence_length, column2=None, bertweettokenizer=False):
    input_ids, input_masks, input_segments = [], [], []

    for i in tqdm(range(df.shape[0])):
        t = df[column1].iloc[i]

        if column2:
            ids, masks, segments = _convert_to_transformer_inputs(t, tokenizer, max_sequence_length, text2=df[column2].iloc[i], bertweettokenizer=bertweettokenizer)
        else:
            ids, masks, segments = _convert_to_transformer_inputs(t, tokenizer, max_sequence_length, bertweettokenizer=bertweettokenizer)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    
    #return np.asarray(input_ids, dtype=np.int32)


    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

def compute_lstm_input_arrays(df, column, max_sequence_length, MAX_NB_WORDS=50000, tokenizer=None):
    if tokenizer:
        input_ids = tokenizer.texts_to_sequences(df[column])
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(maxlen=max_sequence_length, sequences=input_ids, padding='post', value=0)
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='UNK', num_words=MAX_NB_WORDS+1)
        tokenizer.fit_on_texts(df[column])
        tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS+1}

        input_ids = tokenizer.texts_to_sequences(df[column])
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(maxlen=max_sequence_length, sequences=input_ids, padding='post', value=0)

    return np.asarray(input_ids), tokenizer

class mnli_data:
    def __init__(self, all_labels):
        self.all_labels = all_labels

    def convert_to_mnli_format(self, id, text, label=None):
        if label:
            ids = [id]
            texts_1 = [text]
            texts_2 = ["this text is {}".format(label)]
            labels = ["entailment"]
            orig_label = [label]
        else:
            ids = []
            texts_1 = []
            texts_2 = []
            labels = []
            orig_label = []

        for l in self.all_labels:
            if label:
                if l != label:
                    ids.append(id)
                    texts_1.append(text)
                    texts_2.append("this text is {}".format(l))
                    labels.append("contradiction")
                    orig_label.append(l)
            else:
                ids.append(id)
                texts_1.append(text)
                texts_2.append("this text is {}".format(l))
                orig_label.append(l)

        return ids, texts_1, texts_2, labels, orig_label
        
def read_text_embeddings(filename):
    embeddings = []
    word2index = {}
    with open(filename, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip().split()
            if len(list(map(float, line[1:]))) > 1:
                if line[0].lower() not in word2index:
                    word2index[line[0].lower()] = i
                    embeddings.append(list(map(float, line[1:])))
    assert len(word2index) == len(embeddings)
    return word2index, np.array(embeddings)
    
def compute_output_arrays(df, column):
    return np.asarray(df[column])

def subword_tokenization(text, tokenizer, max_text_len, max_char_len):
    
    tokenized_text = []
    for word in text.split():
        if len(word) >= max_char_len:
            word_padded = word[:max_char_len] 
        else:
            word_padded = word + ' '*(max_char_len - len(word))
        
        tokenized_text += [np.array(tokenizer.texts_to_sequences(word_padded)).reshape(1,-1)[0]]
    
    if len(tokenized_text) >= max_text_len:
        tokenized_text = tokenized_text[:max_text_len]
    else:
        for i in range(max_text_len-len(tokenized_text)):
            tokenized_text += [[0]*max_char_len]
        
    return np.asarray(tokenized_text)
    
class TorchDataLoader:
    def __init__(self, ids, mask, token_type_ids, targets):
        self.ids = ids
        self.mask = mask
        self.token_type_ids = token_type_ids
        self.targets = targets

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        ids = self.ids[item]
        mask = self.mask[item]
        token_type_ids = self.token_type_ids[item]
        targets = self.targets[item]

        return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "targets": torch.tensor(targets, dtype=torch.float)
                }

