
import tensorflow as tf
import numpy as np

def create_embedding_matrix(model):
    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def tf_model(embedding_matrix, wv):
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # embedding layer weights are frozen to avoid updating embeddings while training
    saved_embeddings = tf.constant(embedding_matrix)
    embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

    # create the cosine similarity operations
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = embedding / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # call our similarity operation
        sim = similarity.eval()
        # run through each valid example, finding closest words
        for i in range(valid_size):
            valid_word = wv.index2word[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = wv.index2word[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)



if __name__ == "__main__":
    model = gensim.models.Word2Vec.load(root_path + "mymodel")
    embedding_matrix = create_embedding_matrix(model)
    tf_model(embedding_matrix, model.wv)
