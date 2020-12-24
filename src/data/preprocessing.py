import re
import nltk
import spacy
from sklearn import feature_extraction 

def clean_tweets(text):
    text = text.lower()
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'http\w+','',text)
    text = re.sub(r'#\w+','',text)
    text = re.sub(r'\d+','',text)
    return text.strip()

def remove_html(text):
    text = text.replace("\n"," ")
    pattern = re.compile('<.*?>') #all the HTML tags
    return pattern.sub(r'', text)

def remove_email(text):
    text = re.sub(r'[\w.<>]*\w+@\w+[\w.<>]*', " ", text)
    return text

def remove_all_special_chars(text):
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    return text

def replace_mult_spaces(text):
    text = text.replace("&quot","")
    pattern = re.compile(' +')
    text = pattern.sub(r' ', text)
    text = text.strip()
    return text

def replace_chars(text,pattern):
    '''
    e.g.
    pattern = '[()!@&;]'
    '''
    pattern = re.compile(pattern)
    text = pattern.sub(r'', text)

    return text

def nltk_lemmatizer(text,tokenizer=nltk.tokenize.WordPunctTokenizer().tokenize,stemmer=nltk.stem.WordNetLemmatizer()):
    '''
    tokenizer supports NLTK tokenizer, spacy tokenizer, huggingface tokenizer
    NLTK -> use nltk.tokenize.WordPunctTokenizer().tokenize
    Spacy -> spacy_tokenizer('en_core_web_sm')
    Huggingface -> transformers.BertTokenizer.from_pretrained('bert-base-uncased').tokenize

    stemmer supports NLTK stemmers and lemmatizers
    '''

    try:
        text = ' '.join([stemmer.stem(word) for word in tokenizer(text)])
    except AttributeError:
        text = ' '.join([stemmer.lemmatize(word) for word in tokenizer(text)])

    return text

def tokenize_text(text,tokenizer):
    '''
    tokenizer supports NLTK tokenizer, spacy tokenizer, huggingface tokenizer
    NLTK -> use nltk.tokenize.WordPunctTokenizer().tokenize
    Spacy -> spacy_tokenizer('en_core_web_sm')
    Huggingface -> transformers.BertTokenizer.from_pretrained('bert-base-uncased').tokenize
    Also supports custom wordpiece tokenizer
    '''

    if type(text) == list:
        return text #already tokenized
    else:
        return tokenizer(text)

def spacy_lemmatizer(text,spacy_dictonary):
    nlp = spacy.load(spacy_dictonary, parse=True, tag=True, entity=True)
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def stopword_removal(text):
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    stopword_list = nltk.corpus.stopwords.words('english')
    text = nlp(text)
    tokens = [word.text for word in text]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

class spacy_tokenizer:
    def __init__(self,dictionary='en_core_web_sm'):
        self.nlp = spacy.load(dictionary)

    def __call__(self,text):
        return [i.text for i in self.nlp(text)]

def vectorizer(corpus,max_df=.7,min_df=2,max_features=10000,vectorizer_type='tfidf',tokenizer=None, \
                analyzer='word',lowercase=True,stop_words='english',ngram_range=(1,1)):
    '''
    tokenizer supports NLTK tokenizer, spacy tokenizer, huggingface tokenizer
    NLTK -> use nltk.tokenize.WordPunctTokenizer().tokenize
    Spacy -> spacy_tokenizer('en_core_web_sm')
    Huggingface -> transformers.BertTokenizer.from_pretrained('bert-base-uncased').tokenize
    '''

    if type(corpus[0]) == list:
        corpus = [" ".join(i) for i in corpus]

    if vectorizer_type == 'count':
        vector = feature_extraction.text.CountVectorizer(lowercase=lowercase, #this will convert all the tokens into lower case
                                 stop_words=stop_words, #remove english stopwords from vocabulary. if we need the stopwords this value should be None,
                                 tokenizer=tokenizer,
                                 analyzer=analyzer, #tokens should be words. we can also use char for character tokens
                                 max_features=max_features, #maximum vocabulary size to restrict too many features
                                 max_df=max_df, #if some word is in more than 50% of the documents, remove them
                                 min_df=min_df, #words need to be in atleast 2 documents,
                                 ngram_range=ngram_range
                                )
    else:
        vector = feature_extraction.text.TfidfVectorizer(lowercase=lowercase, #this will convert all the tokens into lower case
                                 stop_words=stop_words, #remove english stopwords from vocabulary. if we need the stopwords this value should be None
                                 tokenizer=tokenizer,
                                 analyzer=analyzer, #tokens should be words. we can also use char for character tokens
                                 max_features=max_features, #maximum vocabulary size to restrict too many features
                                 max_df=max_df, #if some word is in more than 50% of the documents, remove them
                                 min_df=min_df, #words need to be in atleast 2 documents
                                 ngram_range=ngram_range
                                )

    vectorized_corpus = vector.fit_transform(corpus)

    return vectorized_corpus, vector