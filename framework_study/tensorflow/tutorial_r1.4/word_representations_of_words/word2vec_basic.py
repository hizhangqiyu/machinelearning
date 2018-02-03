import tensorflow as tf
import collections
import math
import os, random
from tmepfile import gettempdir
import zipfile
import numpy as np 
from six.moves import urllib


url = 'http://mattmahoney.net/dc/'

VOCABULARY_SIZE = 50000
BATCH_SIZE = 128
EMBEDDING_SIZE = 128
SKIP_WINDOW = 1
NUM_SKIPS = 2
NUM_SAMPLED = 64
VALID_SIZE = 16
VALID_WINDOW = 100
NUM_STEPS = 100001

valid_examples = np.random.choice(VALID_WINDOW, VALID_SIZE, replace=False)

def maybe_download(filename, expected_bytes):
    local_filename = os.path.join('../../../dataset/vocabulary', filename)
    if not os.path.exists(local_filename):
        local_filename, - = urllib.request.urlretrieve(url + filename, local_filename)
    
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %r', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verfied ' + local_filename + '. Can you get to it with a browser?')

    return local_filename

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, n_words):
    # ((word, count), ...)
    count = [['UNK', -1]]
    # only use top n_words-1 most common words.
    count.extend(collections.Counter(words).most_common(n_words - 1))
    
    # ((word, index), ...)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # (index, ...)
    data = list()
    unk_count = 0
    for word in words:
        indx = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data_index = 0
def generate_batch(data, batch_size, num_skips, skip_window):
    '''
        num_skips:      number of target in one span
        skip_window:    number of context words
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = 2 * skip_window + 1 # (ctx_word0, ctx_word1, target, ctx_word2, ctx_word3)
    buffer = collections.deque(maxlen=span)
    
    if data_index + span > len(data):
        data_index = 0         # round back to begin

    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)

        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]

        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index = 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

filename = maybe_download('text8.zip', 31344016)
vocabulary = read_data(filename)
data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, VOCABULARY_SIZE)
del vocabulary # save memory usage
print('Top 5 most common words (+UNK) %r', count[:5])

batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

with tf.Graph().as_default(), tf.Session() as session::
    train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    train_labels = tf.palceholder(tf.int32, shape=[BATCH_SIZE, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Varibale(tf.random_uniform(
            [VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Varibale(tf.truncated_normal(
            [VOCABULARY_SIZE, EMBEDDING_SIZE], stddev=1.0 / match.sqrt(EMBEDDING_SIZE)))

        nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                        biases=nce_biases,
                                        labels=train_labels,
                                        inputs=train_inputs,
                                        num_sampled=num_sampled,
                                        num_classes=VOCABULARY_SIZE))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init.run()
    average_loss = 0
    for step in range(NUM_STEPS):
        batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step %r: %r' % (step, average_loss))
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        
    plt.savefig(filename)

try:
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt
    
      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
      plot_only = 500
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
      labels = [reverse_dictionary[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
