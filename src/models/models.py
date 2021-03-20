import tensorflow as tf
import numpy as np
from .layers import *
from transformers import TFAutoModel, AutoTokenizer, AutoConfig

def WLSTM(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None, \
         max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)
    
    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def CLSTM(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None, \
         max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_char_len)(char_inputs)
    
    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model


def CMSA(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None, kernel_size=3, \
         max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_char_len)(char_inputs)
    
    emb = tf.keras.layers.Conv1D(filters=n_units,kernel_size=kernel_size,strides=1,padding='valid',
                            activation='relu')(emb)
    emb = tf.keras.layers.MaxPooling1D(kernel_size)(emb)
    
    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        attention = AttentionWithContext(name='attention')(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, attention, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, attention]))
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    dense = tf.keras.layers.Dense(n_units//2)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    dense = tf.keras.layers.Dense(n_units//4)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    dense = tf.keras.layers.Dense(n_units//8)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model


def HAN(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None, \
        max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)
    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_word_char_len)(char_inputs)
    
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    for i in range(n_layers-1):
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)

    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_units))(lstm)
    dense = AttentionWithContext(name='char_attention')(dense)
    
    char_model = tf.keras.models.Model(char_inputs, dense)
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    word_encoder = tf.keras.layers.TimeDistributed(char_model)(subword_inputs)
    
    #word_embedding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    #word_encoder = word_encoder + word_embedding

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_encoder)
    for i in range(n_layers-1):
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)

    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_units))(lstm)
    
    if seq_output == False:
        dense = AttentionWithContext(name='word_attention')(dense)
    else:
        dense = Attention(n_units,name='word_attention')(dense)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([dense, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def CS_ELMO(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None, \
            max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim == n_units
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_word_char_len)(subword_inputs)
    
    bigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=2,strides=1,padding='same',
                            activation='relu'))(emb)
    
    bigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    bigram_position = bigram_embedding_layer(subword_inputs)
    bigram_position = bigram_emb + bigram_position

    bigram_emb = tf.keras.layers.TimeDistributed(MultiHeadSelfAttention(2*n_units))(tf.keras.layers.Concatenate()([bigram_emb,bigram_position]))
    
    trigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=3,strides=1,padding='same',
                            activation='relu'))(emb)
    
    trigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    trigram_position = trigram_embedding_layer(subword_inputs)
    trigram_position = trigram_emb + trigram_position

    trigram_emb = tf.keras.layers.TimeDistributed(MultiHeadSelfAttention(2*n_units))(tf.keras.layers.Concatenate()([trigram_emb,trigram_position]))
    
    word_emb = tf.keras.layers.TimeDistributed(AttentionWithContext())(tf.keras.layers.Concatenate()([bigram_emb,trigram_emb]))
    word_emb = tf.keras.layers.Dense(n_units)(word_emb)
    word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    word_emb = word_emb + word_embbeding

    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(word_emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm,tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def CS_ELMO_without_words(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None, \
            max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim == n_units
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_word_char_len)(subword_inputs)
    
    bigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=2,strides=1,padding='same',
                            activation='relu'))(emb)
    
    bigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    bigram_position = bigram_embedding_layer(subword_inputs)
    bigram_position = bigram_emb + bigram_position

    bigram_emb = tf.keras.layers.TimeDistributed(MultiHeadSelfAttention(2*n_units))(tf.keras.layers.Concatenate()([bigram_emb,bigram_position]))
    
    trigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=3,strides=1,padding='same',
                            activation='relu'))(emb)
    
    trigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    trigram_position = trigram_embedding_layer(subword_inputs)
    trigram_position = trigram_emb + trigram_position

    trigram_emb = tf.keras.layers.TimeDistributed(MultiHeadSelfAttention(2*n_units))(tf.keras.layers.Concatenate()([trigram_emb,trigram_position]))
    
    word_emb = tf.keras.layers.TimeDistributed(AttentionWithContext())(tf.keras.layers.Concatenate()([bigram_emb,trigram_emb]))
    word_emb = tf.keras.layers.Dense(n_units)(word_emb)
    #word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    #word_emb = word_emb + word_embbeding

    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(word_emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm,tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def Transformer(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(max_char_len, wpe_vocab_size, emb_dim)
    x = embedding_layer(wpe_inputs)
    
    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x)
    
    if seq_output == False:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def BERT(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    automodel = TFAutoModel.from_pretrained(transformer_model_pretrained_path)
    x = automodel(wpe_inputs)[1]
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

all_models = {BERT.__name__: BERT, Transformer.__name__: Transformer, CS_ELMO.__name__: CS_ELMO, CS_ELMO_without_words.__name__:CS_ELMO_without_words, HAN.__name__: HAN, \
            CMSA.__name__: CMSA, WLSTM.__name__: WLSTM}
