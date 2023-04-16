import os
import sys
import time
import json
import jieba
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Embedding, LSTM, Bidirectional, Concatenate, Dense, Flatten, Activation, \
    RepeatVector, Permute, Multiply
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model, load_model

application_path = ''
# 确定存放目录的相对位置
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

# base_dir = os.path.abspath(os.path.join(application_path, os.path.pardir))
base_dir = application_path
train_dir = os.path.join(base_dir, 'train_data')
dict_path = os.path.join(train_dir, 'words_dict.txt')
encoder_input = os.path.join(train_dir, 'encoder_input.npy')
decoder_input = os.path.join(train_dir, 'decoder_input.npy')
decoder_output = os.path.join(train_dir, 'decoder_output.npy')
model_path = os.path.join(train_dir, 's2s.h5py')


def padding_sign(padded_seq, dict_size, mode):
    sign = []
    # decoder_input添加SOS
    if mode:
        for i in range(len(padded_seq)):
            sign.append([dict_size + 1])
        sign = np.array(sign)
        arr = np.concatenate([sign, padded_seq], axis=-1)
    # decoder_target添加EOS
    else:
        for i in range(len(padded_seq)):
            padded_seq[i].append(dict_size + 2)
        arr = padded_seq
    return arr


# 数据预处理
def pre_precess():
    print(time.asctime() + "读取数据")
    # // 读取数据
    df = pd.read_excel("train.xlsx", header=0)
    input_str = df["input"].tolist()
    output_str = df["output"].tolist()

    # 分词
    input_cut = str_cut(input_str)
    output_cut = str_cut(output_str)

    # 使用tensorflow转换为向量
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input_cut + output_cut)

    # 字典
    dict = tokenizer.word_index
    # 保存字典
    with open(dict_path, 'w') as f:
        json.dump(dict, f)

    # 输入向量
    input_vec = tokenizer.texts_to_sequences(input_cut)
    # 输出向量
    output_vec = tokenizer.texts_to_sequences(output_cut)

    # 填充序列至一样的长度，方便获取最长的序列长度
    encoder_input_data = np.array(pad_sequences(input_vec, padding='post'))
    decoder_padding_data = np.array(pad_sequences(output_vec, padding='post'))

    # 统一输入输出序列的长度
    # 获取最长的序列
    padding_len = 0
    if len(encoder_input_data[0]) > len(decoder_padding_data[0] + 1):
        padding_len = len(encoder_input_data[0] + 1)
    else:
        padding_len = len(decoder_padding_data[0] + 1)
        encoder_input_data = np.array(pad_sequences(input_vec, padding='post', maxlen=padding_len))

    decoder_origin = np.array(output_vec, dtype=object)
    decoder_input_data = padding_sign(pad_sequences(decoder_origin, padding='post', maxlen=padding_len - 1),
                                      len(dict), 1)
    decoder_target_data = pad_sequences(padding_sign(decoder_origin, len(dict), 0),
                                        padding='post',
                                        maxlen=padding_len)

    # 保存训练数据
    np.save(encoder_input, encoder_input_data)
    np.save(decoder_input, decoder_input_data)
    np.save(decoder_output, decoder_target_data)
    print(time.asctime() + ' 训练数据处理完成')


# 分词
def str_cut(strs):
    cut = []
    for txt in strs:
        split = jieba.cut(str(txt))
        single = []
        for i in split:
            single.append(i)
        cut.append(single)
    return cut


def get_dict():
    with open(dict_path, 'r') as f:
        emb_dict = json.load(f)
    return emb_dict


# 构建训练模型
def setup_model():
    encoder_input_data = np.load(encoder_input)
    seq_length = len(encoder_input_data[0])

    emb_dict = get_dict()
    # 包括了EOS SOS的长度
    vocabulary_size = len(emb_dict) + 3
    embedding_dim = int(pow(vocabulary_size, 1.0 / 4))+1
    latent_dim = embedding_dim * 40

    print(time.asctime() + ' 词典长度为：' + str(len(emb_dict)))
    print(time.asctime() + ' 拓展后长度为：' + str(embedding_dim))
    # 设置encoder
    # 设置embeddings层
    print(time.asctime() + ' 构建训练模型...')
    encoder_inputs = Input(shape=(None,), name='encoder_input')
    encoder_embedding = Embedding(vocabulary_size,
                                  embedding_dim,
                                  mask_zero=True,
                                  name='encoder_Embedding')(encoder_inputs)
    encoder = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.5),
                            name='encoder_BiLSTM')
    encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = encoder(encoder_embedding)
    state_h = Concatenate(axis=-1, name='encoder_state_h')([fw_state_h, bw_state_h])
    state_c = Concatenate(axis=-1, name='encoder_state_c')([fw_state_c, bw_state_c])
    encoder_states = [state_h, state_c]

    # 设置decoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    decoder_embedding = Embedding(vocabulary_size,
                                  embedding_dim,
                                  mask_zero=True,
                                  name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(latent_dim * 2,
                        return_sequences=True,
                        return_state=True,
                        name='decoder_LSTM',
                        dropout=0.5)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding,
                                         initial_state=encoder_states)

    # attention层
    attention = Dense(1, activation='tanh')(encoder_outputs)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(latent_dim * 2)(attention)
    attention = Permute([2, 1])(attention)
    sent_dense = Multiply()([decoder_outputs, attention])

    # Dense层
    decoder_dense = Dense(vocabulary_size, activation='softmax', name='dense_layer')
    decoder_outputs = decoder_dense(sent_dense)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    # Save model
    model.save(model_path)
    print(time.asctime() + ' 模型生成完成')


def train_model(batch_size, epochs):
    print(time.asctime() + ' 正在处理训练数据...')
    encoder_input_data = np.load(encoder_input)
    decoder_input_data = np.load(decoder_input)
    decoder_target_data = np.load(decoder_output)
    print(time.asctime() + ' 循环轮数:' + str(epochs) + ' batch size:' + str(batch_size))
    model = load_model(model_path)
    print(model.summary())
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs)
    model.save(model_path)


if __name__ == "__main__":
    # 数据预处理
    # pre_precess()
    # 构建模型
    # setup_model()
    # 训练模型
    train_model(20, 50)
