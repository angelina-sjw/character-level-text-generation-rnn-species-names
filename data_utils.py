import numpy as np
import csv
from unidecode import unidecode
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(data_path, column):
    vocabs = []

    with open(data_path) as file:
        data_reader = csv.DictReader(file)
        for line in data_reader:
            vocabs.append(unidecode(line[column]))
    return vocabs

def encode_vocab(vocab):
    transliterated_string = unidecode(''.join(vocab))
    unique_chars = set(transliterated_string.lower())
    unique_chars.update(['<start>', '<end>'])
    unique_chars = sorted(unique_chars)
    char_to_index = {ch: idx for idx, ch in enumerate(unique_chars)}
    index_to_char = np.array(unique_chars)
    
    return unique_chars, char_to_index, index_to_char

def encode_name(name, char_to_index_dict):
    return [char_to_index_dict['<start>']] + [char_to_index_dict[c] for c in name] + [char_to_index_dict['<end>']]

def encode_sequences(sequences):
    unique_chars, char_to_index, index_to_char = encode_vocab(sequences)
    encoded_names = [encode_name(name, char_to_index) for name in sequences]
    max_len = max(len(name) for name in encoded_names)
    sequences = pad_sequences(encoded_names, maxlen=max_len, padding='post')
    return sequences, unique_chars, char_to_index, index_to_char

def decode_name(encoded_sequence, index_to_char):
    decoded_name = ""
    for idx in encoded_sequence[1:]:
        if index_to_char[idx] == "<end>":
            break
        char = index_to_char[idx]
        decoded_name += char
    return decoded_name

def create_input_target(sequence):
    input_seq = sequence[:-1]
    target_seq = sequence[1:]
    return input_seq, target_seq

def train_test_sequence_split(sequences, proportion = 0.005, seed = 42):
    np.random.seed(seed)
    input, target = zip(*[create_input_target(seq) for seq in sequences])
    input = np.array(input)
    target = np.array(target)
    
    total_length = len(input)
    test_size = int(proportion * len(input))
    test_index = np.random.choice(total_length, test_size, replace=False)

    test_input = input[test_index]
    test_target = target[test_index]

    train_index = np.delete(np.arange(total_length), test_index)
    train_input = input[train_index]
    train_target = target[train_index]
    
    return train_input, train_target, test_input, test_target

def truncate(sequences, num_var=10, min_length=2):
    all_truncated_sequences = []

    for seq in sequences:
        truncated_variations = []
        first_zero_index = np.argmax(seq == 0) if 0 in seq else len(seq)
        max_length = first_zero_index

        len_range = range(min_length, max_length)
        if len(len_range) >= num_var:
            trun_lenths = np.random.choice(len_range, num_var, replace=False)
        else:
            trun_lenths = np.random.choice(len_range, num_var, replace=True)

        for length in trun_lenths:
            trunc_seq = seq[:length]
            truncated_variations.append(trunc_seq)

        all_truncated_sequences.extend(truncated_variations)
    
    return np.array(all_truncated_sequences, dtype=object)