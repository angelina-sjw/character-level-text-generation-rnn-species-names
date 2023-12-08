import os, csv
import tensorflow as tf
import numpy as np

from data_utils import load_data, encode_sequences, train_test_sequence_split, truncate, decode_name
from model_utils import create_model, model_loss
from test_utils import generate_text, find_matching_vocab

######### directory and data ############
GOOGLE_COLAB = False
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
_, _, test_input, test_target = train_test_sequence_split(sequences)
truncated_test_input = truncate(test_input)
truncated_test_input = np.tile(truncated_test_input, 5)

vocab_length = len(unique_chars)
embed_dim = 256
rnn_units = 1024
batch_size = 64

checkpoint_dir = os.path.join(directory, 'training_checkpoints')
model = create_model(vocab_length, embed_dim, rnn_units, batch_size)
model.compile(optimizer='adam', loss=model_loss)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).assert_existing_objects_matched()


############ Task: new species names generation ###################

new_names_only = os.path.join(directory, 'experiments/new_names_only_10x.csv')
os.makedirs(os.path.dirname(new_names_only), exist_ok=True)
        
with open(new_names_only, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Start String", "Generated New Vocab", 'Attempts'])

for input in truncated_test_input:
    decoded_input = decode_name(input, index_to_char)
    unique_name_found = False
    attempts = 0
    max_attempts = 20 
    while not unique_name_found and attempts < max_attempts:
        generated_vocab = generate_text(model, char_to_index, index_to_char, decoded_input)
        if generated_vocab.lower() not in species_names:
            unique_name_found = True
            with open(new_names_only, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([decoded_input, generated_vocab, attempts])
        attempts += 1

    if not unique_name_found:
        with open(new_names_only, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([decoded_input, '', float('NaN')])

print("New names only generation completed and saved to CSV.")

############ Task: species names completion ###################

complete_names = os.path.join(directory, 'experiments/complete_names_10x.csv')
os.makedirs(os.path.dirname(complete_names), exist_ok=True)
with open(complete_names, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Start String", "Matched Vocab", 'Attempts'])
    
for _ in truncated_test_input:
    input = decode_name(_, index_to_char)
    matched_name, count = find_matching_vocab(model, char_to_index, index_to_char, 
                                              species_names, input)
    matched_name = matched_name[0] if matched_name else ''
    count = count[0] if count else float('NaN')
    with open(complete_names, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input, matched_name, count])

print("Species names completion completed and saved to CSV.")