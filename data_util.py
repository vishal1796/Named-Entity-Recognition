import os
import re
import codecs
import math
import numpy
from keras.preprocessing.sequence import pad_sequences


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = re.sub('\d', '0', line.rstrip())
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences):
    """
    Check and update sentences tagging scheme to IOBES.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for index, sentence in enumerate(sentences):
        tags = [w[-1] for w in sentence]
        
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %d:\n%s' % (index, sentence))
            if split[0] == 'B':
                continue
            elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
            elif tags[i - 1][1:] == tag[1:]:
                continue
            else:  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
                                
        for word, new_tag in zip(sentence, tags):
            word[-1] = new_tag


def build_vocab(documentlist):
    """
    Create a dictionary of items from a list of list of items.
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    assert type(documentlist) is list
    vocab = {}
    for document in documentlist:
        for item in document:
            if item not in vocab:
                vocab[item] = 1
            else:
                vocab[item] += 1
    vocab['<UNK>'] = 10000000  
    
    sorted_vocab = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    id_to_word = {i: v[0] for i, v in enumerate(sorted_vocab)}
    word_to_id = {v: k for k, v in id_to_word.items()}
    return word_to_id


def word_mapping(sentences, lower=False):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    word_to_id = build_vocab(words)
    print "Found %i unique words (%i in total)" % (len(word_to_id), sum(len(x) for x in words))
    return word_to_id

def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    char_to_id = build_vocab(chars)
    print "Found %i unique characters" % len(char_to_id)
    return char_to_id

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    tag_to_id = build_vocab(tags)
    print "Found %i unique named entity tags" % len(tag_to_id)
    return tag_to_id

def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>'] for w in str_words]
        chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })
    return data


def build_X_Y(dataSet, max_words=50, maxCharLength=20):
    word_id = []
    tag_id = []
    char = []
    sentencelist = []
    char_sequence = []
    
    for row in dataSet:
        sentence = row['str_words']
        sentence = sentence[:max_words]
        sentence += ['<UNK>'] * (max_words - len(sentence)) 
        sentencelist.append(sentence)
        
    for row in dataSet:
        wordlist = row['chars']
        sentence = []
        for word in wordlist[:max_words]:
            word = word[:maxCharLength]
            word += [0]*(maxCharLength -len(word))
            sentence += word
        sentence += [0] * maxCharLength * (max_words- len(wordlist))
        char_sequence.append(sentence)
        
    for row in dataSet:
        word_id.append(row['words'])
        tag_id.append(row['tags'])
        
    word_id = pad_sequences(word_id, maxlen=max_words, padding='post', truncating='post', value=0)
    tag_id = pad_sequences(tag_id, maxlen=max_words, padding='post', truncating='post', value=0)   
    
    return sentencelist, numpy.array(char_sequence), numpy.array(word_id), numpy.array(tag_id)


def build_embeddingmatrix(word_to_id, word_embedding_dim=100):    
    embeddings_index = {}
    f = open(os.path.join('/glove/', 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = numpy.zeros((len(word_to_id) + 1, word_embedding_dim))
    for word, i in word_to_id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix   