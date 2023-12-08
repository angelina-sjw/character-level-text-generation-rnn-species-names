import tensorflow as tf

def create_model(vocab_len, embed_dim, rnn_units, batch_sz, dropout_rate=0.2):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_len, embed_dim, 
                                  mask_zero=True, batch_input_shape=[batch_sz, None]),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, 
                            stateful=True, recurrent_initializer='glorot_uniform', 
                            recurrent_dropout=dropout_rate),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(vocab_len)
    ])
    
    return model

def model_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def get_last_epoch(checkpoint_dir):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        last_epoch = int(latest_checkpoint.split('_')[-1])
        return last_epoch
    return 0