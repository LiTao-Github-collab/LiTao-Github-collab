import re
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

# 设置随机种子以确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

# 读取数据
def read_data():
    with open("english.txt", "r", encoding="utf-8") as f:
        english_text = f.read().splitlines()
    with open("french.txt", "r", encoding="utf-8") as f:
        french_text = f.read().splitlines()
    return english_text, french_text

# 预处理数据
def preprocess_data(english_text, french_text):
    # 添加起始和结束标记
    english_text = ["<start> " + line + " <end>" for line in english_text]
    french_text = ["<start> " + line + " <end>" for line in french_text]

    # 创建输入和目标数据集
    X_train, X_val, y_train, y_val = train_test_split(english_text, french_text, test_size=0.2, random_state=None)

    # 创建英文字典
    #english_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    #english_tokenizer.fit_on_texts(X_train)
    #english_vocab_size = len(english_tokenizer.word_index) + 1
    english_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
    english_tokenizer.fit_on_texts(X_train)

    # 生成词汇表后，添加起始标记和结束标记
    english_tokenizer.word_index['<start>'] = len(english_tokenizer.word_index) + 1
    english_tokenizer.word_index['<end>'] = len(english_tokenizer.word_index) + 1

    # 创建法文字典
    french_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    french_tokenizer.fit_on_texts(y_train)
    french_vocab_size = len(french_tokenizer.word_index) + 1

    # 将文本序列转换为数字序列
    X_train = english_tokenizer.texts_to_sequences(X_train)
    X_val = english_tokenizer.texts_to_sequences(X_val)
    y_train = french_tokenizer.texts_to_sequences(y_train)
    y_val = french_tokenizer.texts_to_sequences(y_val)

    # 填充序列，使其具有相同的长度
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post')
    X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val, padding='post')
    y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train, padding='post')
    y_val = tf.keras.preprocessing.sequence.pad_sequences(y_val, padding='post')

    return X_train, X_val, y_train, y_val, english_vocab_size, french_vocab_size, english_tokenizer, french_tokenizer

# 创建位置编码
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# 创建遮挡
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

# 创建前瞻遮挡（用于解码器中的自注意力）
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

# 构建多头自注意力层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

# 创建位置前馈网络层
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

# 构建编码器层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# 构建解码器层
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

# 构建编码器
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

# 构建解码器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights

# 构建transformer模型
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

english_vocab_size = 20000
french_vocab_size = 20000
MAX_LENGTH_INPUT = 1000
MAX_LENGTH_OUTPUT = 1000

# 设置超参数
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = english_vocab_size
target_vocab_size = french_vocab_size
dropout_rate = 0.1

# 创建模型实例
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=input_vocab_size, pe_target=target_vocab_size, rate=dropout_rate)

# 定义CustomSchedule类
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)  # 转换为浮点数张量

    def __call__(self, step):
        return tf.math.rsqrt(tf.cast(step, tf.float32))  # 转换为浮点数张量

# 设置超参数
d_model = 128

# 定义优化器
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# 定义损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# 定义评估指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# 创建检查点
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 恢复最新的检查点（如果存在）
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!')

# 训练模型
EPOCHS = 10

def create_masks(inp, tar):
    # Encoder的padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Decoder的padding mask
    dec_padding_mask = create_padding_mask(inp)

    # Decoder的look ahead mask
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到注意力权重
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

# 加载数据
english_text, french_text = read_data()
X_train, X_val, y_train, y_val, english_vocab_size, french_vocab_size, english_tokenizer, french_tokenizer = preprocess_data(english_text, french_text)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
train_dataset = train_dataset.batch(64, drop_remainder=True)

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

print('Training finished!')

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

# 使用训练好的模型进行翻译
def translate(sentence):
    sentence = preprocess_sentence(sentence)
    inputs = [english_tokenizer.word_index[i] for i in sentence.split(' ')]

    if len(inputs) > MAX_LENGTH_INPUT:
        return "对不起，您的输入句子太长了，无法处理。请缩短句子长度重新尝试。"

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=MAX_LENGTH_INPUT, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, inputs)

    predictions, attention_weights = transformer(inputs, inputs, False, enc_padding_mask, combined_mask, dec_padding_mask)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    while predicted_id != french_tokenizer.word_index['<end>']:
        if len(result.split(' ')) >= MAX_LENGTH_OUTPUT:
            return "对不起，生成的翻译结果过长，无法继续生成。请缩短输入句子或调整模型参数。"

        result += french_tokenizer.index_word.get(predicted_id.numpy().item(), '') + ' '

        inputs = tf.concat([inputs, predicted_id], axis=-1)

        predictions, attention_weights = transformer(inputs, inputs, False, enc_padding_mask, combined_mask, dec_padding_mask)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.expand_dims(predicted_id, axis=-1)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1)[:, -1], tf.int32)

    return result

# 测试翻译功能
input_sentence = "the clouds floating in the sky are beautiful."
input_tensor = tf.constant([english_tokenizer.word_index["<start>"] + english_tokenizer.word_index[word] for word in input_sentence.split()] + [english_tokenizer.word_index["<end>"]], dtype=tf.int32)
mask = tf.expand_dims(tf.cast(tf.math.equal(input_tensor, 0), tf.float32), axis=1)
translation = translate(input_tensor, mask)

print('Input: {}'.format(input_sentence))
print('Translation: {}'.format(translation))
