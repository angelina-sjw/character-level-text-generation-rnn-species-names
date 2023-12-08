import tensorflow as tf
batch_size = 64

def generate_text(model, char_to_index, index_to_char, start_string=None, temperature=1.0, max_len=20):
    if start_string is None:
        start_string = '<start>'
        input_eval = [char_to_index[start_string]]
    else:
        input_eval = [char_to_index[s] for s in start_string]

    input_eval = tf.expand_dims(input_eval, 0)
    input_eval = tf.repeat(input_eval, batch_size, axis=0)

    text_generated = []
    model.reset_states()
    for i in range(len(start_string) - 1):
        model(input_eval)

    while True:
        predictions = model(input_eval)
        predictions = predictions[0, -1, :] / temperature
        predicted_id = tf.random.categorical(tf.expand_dims(predictions, 0), num_samples=1)[-1,0].numpy()
        if predicted_id == char_to_index['<end>'] or len(text_generated) >= max_len:
            break

        text_generated.append(index_to_char[predicted_id])
        input_eval = tf.repeat(tf.expand_dims([predicted_id], 0), batch_size, axis=0)

    return (start_string + ''.join(text_generated)).replace('<start>', '').replace('<end>', '')

def find_matching_vocab(model, char_to_index, index_to_char, corpus, start_string, max_attempt = 20, output_num = 1, temperature=1.0):
    matched_vocab = []
    attempts_count = []

    for _ in range(output_num):
        attempt = 0
        while attempt < max_attempt:
            attempt += 1
            generated_vocab = generate_text(model, char_to_index, index_to_char, start_string, temperature)
            if generated_vocab in corpus:
                matched_vocab.append(generated_vocab)
                attempts_count.append(attempt)
                break
        if attempt >= max_attempt:
            break

    return matched_vocab, attempts_count

def type_token_ratio(vocabs):
    unique_vocabs = set(vocabs)
    total_vocabs = len(vocabs)
    return len(unique_vocabs) / total_vocabs if total_vocabs > 0 else 0

def ngram_diversity(vocabs, n=2):
    ngrams = set()
    total_ngrams = 0
    for vocab in vocabs:
        for i in range(len(vocab) - n + 1):
            ngrams.add(vocab[i:i+n])
            total_ngrams += 1
    return len(ngrams) / total_ngrams if total_ngrams > 0 else 0

def unique_word_count(vocabs):
    return len(set(vocabs))

def novelty_comparison(generated_vocabs, known_vocabs):
    novel_vocabs = set(generated_vocabs) - set(known_vocabs)
    return len(novel_vocabs) / len(generated_vocabs) if generated_vocabs else 0