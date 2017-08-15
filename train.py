import cPickle as pickle
import numpy as np
from data_util import *
from crf import CRF
from keras.layers import Input
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def my_init(shape, dtype=None):
    scale = np.sqrt(3.0 / 30)
    return K.random_uniform(shape, minval=-scale, maxval=scale, dtype=dtype)


train_sentences = load_sentences('data/eng.train')
valid_sentences = load_sentences('data/eng.testa')

update_tag_scheme(train_sentences)
update_tag_scheme(valid_sentences)

word_to_id = word_mapping(train_sentences + valid_sentences)
embedding_matrix = build_embeddingmatrix(word_to_id)
char_to_id, tag_to_id = char_mapping(train_sentences), tag_mapping(train_sentences)

train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id)
valid_data = prepare_dataset(valid_sentences, word_to_id, char_to_id, tag_to_id)

print "%i / %i  sentences in train / valid." % (len(train_data), len(valid_data))

sentencelist_train, char_sequence_train, word_sequence_train, tag_sequence_train = build_X_Y(train_data)
sentencelist_valid, char_sequence_valid, word_sequence_valid, tag_sequence_valid = build_X_Y(valid_data)

char_embedding_dim = 30
word_embedding_dim = 100
lstm_dim = 200
max_words = 50
maxCharSize = 20 
word_vocab_size = len(word_to_id)
char_vocab_size = len(char_to_id)
tag_label_size = len(tag_to_id)

tags_train = []
tags_valid = []

for i in range(len(tag_sequence_train)):
    tags_train.append(to_categorical(tag_sequence_train[i], tag_label_size))
for i in range(len(tag_sequence_valid)):
    tags_valid.append(to_categorical(tag_sequence_valid[i], tag_label_size))
    
tags_train = np.array(tags_train)
tags_valid = np.array(tags_valid)

print('char_sequence_train shape:', char_sequence_train.shape)
print('word_sequence_train shape:', word_sequence_train.shape)
print('tag_sequence_train shape:', tag_sequence_train.shape)
print('char_sequence_valid shape:', char_sequence_valid.shape)
print('word_sequence_valid shape:', word_sequence_valid.shape)
print('tag_sequence_valid shape:', tag_sequence_valid.shape)

pickle.dump(word_to_id, open("output/word_to_id.pkl", 'wb'))
pickle.dump(char_to_id, open("output/char_to_id.pkl", 'wb'))
pickle.dump(tag_to_id, open("output/tag_to_id.pkl", 'wb'))

print('Train...')
char_input = Input(shape=(maxCharSize * max_words,), dtype='int32', name='char_input')
char_emb = Embedding(char_vocab_size, char_embedding_dim, embeddings_initializer=my_init, input_length = max_words*maxCharSize, name='char_emb')(char_input)
char_emb = Dropout(0.5)(char_emb)
char_cnn = Conv1D(filters=30, kernel_size=3, activation='relu', padding='same')(char_emb)
char_max_pooling = MaxPooling1D(pool_size=maxCharSize)(char_cnn)
word_input = Input(shape=(max_words,), dtype='int32', name='word_input')
word_emb = Embedding(word_vocab_size+1, word_embedding_dim, weights=[embedding_matrix], name='word_emb')(word_input)
final_emb = Concatenate(axis=2, name='final_emb')([word_emb, char_max_pooling])
emb_droput = Dropout(0.5)(final_emb)
bilstm_word = Bidirectional(LSTM(200, kernel_initializer='glorot_uniform', return_sequences=True, unit_forget_bias=True))(emb_droput)
bilstm_word_d = Dropout(0.5)(bilstm_word)
crf = CRF(tag_label_size, learn_mode='marginal', sparse_target=False, use_boundary = False)
crf_output = crf(bilstm_word_d)

model = Model(inputs=[char_input, word_input], outputs=crf_output)

sgd = SGD(lr=0.015, decay=0.05, momentum=0.9, nesterov=False, clipvalue=5)
model.compile(loss=crf.loss_function, optimizer=sgd, metrics=[crf.accuracy])

filepath = "output/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list=[checkpoint]

model.fit([char_sequence_train, word_sequence_train], tags_train,
           validation_data=([char_sequence_valid, word_sequence_valid], tags_valid),
            batch_size=10, epochs=50, callbacks=callbacks_list)

model_json = model.to_json()
with open("output/model_json.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("output/full_model_weight.h5")

model.save('output/bilstm-cnn-crf.h5')