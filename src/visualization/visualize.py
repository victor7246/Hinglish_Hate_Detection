import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data import data_utils

import pandas as pd
import numpy as np
import pandas_bokeh
import matplotlib.pyplot as plt
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from spacy import displacy
import pyLDAvis
import pyLDAvis.sklearn
import torch
import html

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

default_vectorizer = TfidfVectorizer(lowercase=True, #this will convert all the tokens into lower case
                         stop_words='english', #remove english stopwords from vocabulary. if we need the stopwords this value should be None
                         analyzer='word', #tokens should be words. we can also use char for character tokens
                         max_features=10000, #maximum vocabulary size to restrict too many features
                         min_df = 2,
                         max_df = .4,
                         token_pattern=r'[a-z][a-z]+'
                        )

def html_escape(text):
    return html.escape(text)

def plot_top_words(texts, vectorizer=default_vectorizer,topk=20, interactive=False, save_path='../visualizations/'):
    if type(texts[0]) == list:
        texts = [" ".join(LemmaTokenizer()(" ".join(i))) for i in texts]
    else:
        texts = [" ".join(LemmaTokenizer()(i)) for i in texts]

    vectorized_corpus = vectorizer.fit_transform(texts)
    tfidf_scores = vectorized_corpus.toarray().mean(axis=0)
    top_words = vectorizer.get_feature_names()
    top_words_df = pd.DataFrame()

    top_words_df['top_word'] = top_words
    top_words_df['word_weight'] = tfidf_scores #np.exp(tfidf_scores)/np.exp(tfidf_scores).sum()

    top_words_df = top_words_df.sort_values(by=['word_weight'],ascending=[True]).reset_index(drop=True)
    top_words_df = top_words_df.tail(topk).reset_index(drop=True).set_index('top_word')

    if interactive:
        pd.set_option('plotting.backend', 'pandas_bokeh')
        
        html_plot = top_words_df.plot_bokeh(
            kind="barh",
            title="Top Words",
            legend=False,
            show_figure=False,
            return_html=True,
            xlabel="Weight",
            ylabel="Word"
            )

        #Export the HTML string to an external HTML file and show it:
        with open(os.path.join(save_path,"top_words.html") , "w") as f:
            f.write(html_plot)
            
    else:
        pd.set_option('plotting.backend', 'matplotlib')
        plt.figure(figsize=(6,8))
        plt.tight_layout()
        top_words_df.plot.barh(legend=False)
        #plt.legend(False)
        plt.xlabel("Weight")
        plt.ylabel("Word")
        plt.yticks(rotation=30)
        plt.title("Top Words")
        plt.savefig(os.path.join(save_path,"top_words.png"),dpi=200, bbox_inches='tight')
        plt.close()

def plot_top_words_conditional(texts, labels, vectorizer=default_vectorizer,topk=20, interactive=False, save_path='../visualizations/'):
    if type(texts[0]) == list:
        texts = [" ".join(LemmaTokenizer()(" ".join(i))) for i in texts]
    else:
        texts = [" ".join(LemmaTokenizer()(i)) for i in texts]

    vectorized_corpus = vectorizer.fit_transform(texts)
    top_words = vectorizer.get_feature_names()

    top_words_df = pd.DataFrame(vectorized_corpus.toarray(),columns=top_words)
    top_words_df['label'] = labels

    top_words_df = top_words_df.groupby(['label']).mean().reset_index(drop=False)
    #print (top_words_df.head(5))

    #top_words_df = pd.melt(top_words_df, id_vars=['label'],value_vars=list(top_words))
    top_words_df = pd.melt(top_words_df, col_level=0, id_vars=['label'],value_vars=top_words_df.columns[1:])
    top_words_df.columns = ['label','top_word','word_weight']

    top_words_df = top_words_df.sort_values(by=['label','word_weight'],ascending=[True,True]).reset_index(drop=True)
    top_words_df = top_words_df.groupby(['label']).tail(topk).reset_index(drop=True)

    if interactive:
        pd.set_option('plotting.backend', 'pandas_bokeh')

        html_plots = []

        for lb in top_words_df.label.unique():

            html_plots.append(top_words_df[top_words_df['label'] == lb][['top_word','word_weight']].set_index('top_word').plot_bokeh(
                    kind="barh",
                    title="Top Words: Label = {}".format(lb),
                    legend=False,
                    show_figure=False,
                    xlabel="Weight",
                    ylabel="Word"
                    ))
        html_plot = pandas_bokeh.plot_grid([html_plots], 
                       plot_width=450,return_html=True,show_plot=False)


        #Export the HTML string to an external HTML file and show it:
        with open(os.path.join(save_path,"top_words_conditional.html") , "w") as f:
            f.write(html_plot)

    else:
        pd.set_option('plotting.backend', 'matplotlib')
        fig, ax = plt.subplots(1,top_words_df.label.nunique(),figsize=(15,6))
        fig.tight_layout()
        for i, lb in enumerate(top_words_df.label.unique()):
            top_words_df[top_words_df['label'] == lb][['top_word','word_weight']].set_index('top_word').plot.barh(legend=False, ax=ax[i])
            ax[i].set_xlabel("Weight")
            if i == 0:
                ax[i].set_ylabel("Word")
            else:
                ax[i].set_ylabel(None)
            ax[i].set_title("Top Words: Label = {}".format(lb))

        #plt.title("Top Words")
        plt.savefig(os.path.join(save_path,"top_words_conditional.png"),dpi=200, bbox_inches='tight')
        plt.close()

def spacy_dependency_graph(text, spacy_dictionary='en_core_web_sm',save_path='../visualizations/'):
    if type(text) == list:
        text = " ".join(text)

    nlp = spacy.load(spacy_dictionary)
    doc = nlp(text)
    html = displacy.render(doc,"dep",jupyter=False, options={'distance':100})

    with open(os.path.join(save_path,"dependency_graph.html") , "w") as f:
        f.write(html)

def spacy_entities(text, spacy_dictionary='en_core_web_sm',save_path='../visualizations/'):
    if type(text) == list:
        text = " ".join(text)

    nlp = spacy.load(spacy_dictionary)
    doc = nlp(text)
    html = displacy.render(doc,"ent",jupyter=False)

    with open(os.path.join(save_path,"entities.html") , "w") as f:
        f.write(html)

def plot_topic_models(texts,vectorizer=default_vectorizer,topk=20, num_topics=10, save_path='../visualizations/'):
    if type(texts[0]) == list:
        texts = [" ".join(LemmaTokenizer()(" ".join(i))) for i in texts]
    else:
        texts = [" ".join(LemmaTokenizer()(i)) for i in texts]

    lda = LatentDirichletAllocation(n_components=num_topics)
    vectorized_corpus = vectorizer.fit_transform(texts)
    lda_features = lda.fit_transform(vectorized_corpus)

    viz = pyLDAvis.sklearn.prepare(lda_model=lda,vectorizer=vectorizer,dtm=vectorized_corpus)

    pyLDAvis.save_html(viz,os.path.join(save_path,'topic_model.html'))

def plot_attention_weights(text, base_model, bpetokenizer, save_path='../visualizations/', max_len=0, tokenizer=None):
    if type(text) == list:
        text = " ".join(text)

    d = data_utils.process_data_for_transformers(text,bpetokenizer,tokenizer,0)
    d = {
            "ids": torch.tensor([d['ids']], dtype=torch.long),
            "mask": torch.tensor([d['mask']], dtype=torch.long),
            "token_type_ids": torch.tensor([d['token_type_ids']], dtype=torch.long)
        }

    try:
        orig_tokens = [0] + bpetokenizer.encode(text).ids + [2]
        orig_tokens  = [bpetokenizer.id_to_token(j) for j in orig_tokens]
    except:
        orig_tokens = tokenizer.tokenize(text,add_special_tokens=True)

    base_model.eval()
    attention_weights = base_model(d["ids"],d["mask"],d["token_type_ids"])[-1][-1].detach().numpy()
    attention_weights = attention_weights[0].mean(axis=0)

    fig, ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(attention_weights)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(orig_tokens)))
    ax.set_yticks(np.arange(len(orig_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(orig_tokens)
    ax.set_yticklabels(orig_tokens)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    ax.set_title("Attention Weights")
    fig.tight_layout()
    #plt.xticks(rotation=45)
    plt.title("Attention weights")
    plt.savefig(os.path.join(save_path,"attention_weights.png"),dpi=200, bbox_inches='tight')
    plt.close()


def conditional_weights(text, model, bpetokenizer, max_alpha=3, save_path='../visualizations/', max_len=80, tokenizer=None):
    if type(text) == list:
        text = " ".join(text)

    orig_text = text

    all_texts = [" ".join(text.split()[:i]) for i in range(len(text.split())+1)]

    #print (all_texts)
    #all_texts = [" ".join(text.split()[:i]+text.split()[i+1:]) for i in range(len(text.split())+1)]

    d = {"ids":[],"mask":[],"token_type_ids":[]}

    for text in all_texts:

        d_ = data_utils.process_data_for_transformers(text,bpetokenizer,tokenizer,max_len)
        d["ids"].append(d_["ids"])
        d["mask"].append(d_["mask"])
        d["token_type_ids"].append(d_["token_type_ids"])
        
    d = {
            "ids": torch.tensor(d['ids'], dtype=torch.long),
            "mask": torch.tensor(d['mask'], dtype=torch.long),
            "token_type_ids": torch.tensor(d['token_type_ids'], dtype=torch.long)
        }

    model.eval()
    preds = torch.sigmoid(model(d["ids"],d["mask"],d["token_type_ids"])).detach().numpy()

    weights = []
    for i in range(1,preds.shape[0]):
        weights.append(preds[i,0]/preds[i-1,0])
        
    max_alpha = max_alpha
    highlighted_text = []

    for i, word in enumerate(orig_text.split()):


        weight = weights[i]
        
        if weight is not None:
            highlighted_text.append('<span style="background-color:rgba(135,206,250,' + str(weight / max_alpha) + ');">' + html_escape(word) + '</span>')
        else:
            highlighted_text.append(word)
            
    highlighted_text = ' '.join(highlighted_text)

    with open(os.path.join(save_path,"conditional_weights.html") , "w") as f:
        f.write(highlighted_text)

    return weight

def captum_text_interpreter(text, model, bpetokenizer, idx2label, max_len=80, tokenizer=None, multiclass=False):
    if type(text) == list:
        text = " ".join(text)

    d = data_utils.process_data_for_transformers(text,bpetokenizer,tokenizer,0)
    d = {
            "ids": torch.tensor([d['ids']], dtype=torch.long),
            "mask": torch.tensor([d['mask']], dtype=torch.long),
            "token_type_ids": torch.tensor([d['token_type_ids']], dtype=torch.long)
        }

    try:
        orig_tokens = [0] + bpetokenizer.encode(text).ids + [2]
        orig_tokens  = [bpetokenizer.id_to_token(j) for j in orig_tokens]
    except:
        orig_tokens = tokenizer.tokenize(text,add_special_tokens=True)

    model.eval()
    if multiclass:
    	preds_proba = torch.sigmoid(model(d["ids"],d["mask"],d["token_type_ids"])).detach().cpu().numpy()
    	preds = preds_proba.argmax(-1)
    	preds_proba = preds_proba[0][preds[0][0]]
    	predicted_class = idx2label[preds[0][0]]
    else:
    	preds_proba = torch.sigmoid(model(d["ids"],d["mask"],d["token_type_ids"])).detach().cpu().numpy()
    	preds = np.round(preds_proba)
    	preds_proba = preds_proba[0][0]
    	predicted_class = idx2label[preds[0][0]]

    lig = LayerIntegratedGradients(model, model.base_model.embeddings)
    
    reference_indices = [0] + [1]*(d["ids"].shape[1]-2) + [2]
    reference_indices = torch.tensor([reference_indices], dtype=torch.long)
    
    attributions_ig, delta = lig.attribute(inputs=d["ids"],baselines=reference_indices,additional_forward_args=(d["mask"],d["token_type_ids"]), \
                                           return_convergence_delta=True)
    
    attributions = attributions_ig.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().cpu().numpy()

    visualization.visualize_text([visualization.VisualizationDataRecord(
                            word_attributions=attributions,
                            pred_prob=preds_proba,
                            pred_class=predicted_class,
                            true_class=predicted_class,
                            attr_class=predicted_class,
                            attr_score=attributions.sum(),       
                            raw_input=orig_tokens,
                            convergence_score=delta)])

