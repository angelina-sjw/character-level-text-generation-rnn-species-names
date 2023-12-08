import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt

from data_utils import load_data, encode_sequences, train_test_sequence_split
from model_utils import create_model, model_loss, get_last_epoch

NEW_TRAIN = True
GOOGLE_COLAB = False
EPOCHS = 50

######### directory and data ############

if GOOGLE_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    directory = '/content/drive/My Drive/spider_project'
else:
    directory = os.getcwd() 
data_path = os.path.join(directory, 'data/Spiders.csv')

#########################################
    
species_names = set(load_data(data_path, 'species'))
sequences, unique_chars, char_to_index, index_to_char = encode_sequences(species_names)
train_input, train_target, _, _ = train_test_sequence_split(sequences)

batch_size = 64
buffer_size = 10000
train_data = tf.data.Dataset.from_tensor_slices(
    (train_input, train_target)
    ).shuffle(buffer_size).batch(batch_size, drop_remainder=True)

vocab_length = len(unique_chars)
embed_dim = 256
rnn_units = 1024

checkpoint_dir = os.path.join(directory, 'training_checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
best_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_prefix, 
    save_weights_only=True, 
    save_best_only=True, 
    monitor='loss', 
    mode='min'
)

csv_logger = CSVLogger('training_log.csv', append=True)
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
callbacks = [best_checkpoint_callback, csv_logger, early_stopping]

model = create_model(vocab_length, embed_dim, rnn_units, batch_size)
model.compile(optimizer='adam', loss=model_loss)

if NEW_TRAIN:
    history = model.fit(train_data, epochs=EPOCHS, callbacks=callbacks)
else:
    last_epoch = get_last_epoch(checkpoint_dir)
    if early_stopping.stopped_epoch > 0:
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).assert_existing_objects_matched()
    else:
        checkpoint = tf.train.Checkpoint(model=model)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).assert_existing_objects_matched()
        history = model.fit(train_data, initial_epoch=last_epoch, epochs=EPOCHS, callbacks=callbacks)
        
training_log_path = os.path.join(directory, 'training_log.csv')
training_log = pd.read_csv(training_log_path)

plt.figure(figsize=(10, 6))
plt.plot(training_log['epoch'], training_log['loss'], marker='o', color='b')
plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.grid(True)
plt.savefig(os.path.join(directory, 'training_loss.png'))

print('training completed')
