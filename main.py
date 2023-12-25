import os
import csv
import subprocess
import re
import random
import numpy as np


def read_in_shakespeare():
  '''Reads in the Shakespeare dataset processesit into a list of tuples.
     Also reads in the vocab and play name lists from files.

  Each tuple consists of
  tuple[0]: The name of the play
  tuple[1] A line from the play as a list of tokenized words.

  Returns:
    tuples: A list of tuples in the above format.
    document_names: A list of the plays present in the corpus.
    vocab: A list of all tokens in the vocabulary.
  '''

  tuples = []

  with open('will_play_text.csv') as f:
    csv_reader = csv.reader(f, delimiter=';')
    for row in csv_reader:
      play_name = row[1]
      line = row[5]
      line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
      line_tokens = [token.lower() for token in line_tokens]

      tuples.append((play_name, line_tokens))

  with open('vocab.txt') as f:
    vocab =  [line.strip() for line in f]

  with open('play_names.txt') as f:
    document_names =  [line.strip() for line in f]

  return tuples, document_names, vocab

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))

    # Initialize the term-document matrix with zeros
    td_matrix = np.zeros((len(vocab), len(document_names)))

    for doc_id, (play_name, line_tokens) in enumerate(line_tuples):
        for token in line_tokens:
            if token in vocab_to_id:
                word_id = vocab_to_id[token]
                td_matrix[word_id, doc_id] += 1  # Increment the count for the term in the document

    return td_matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    # Initialize the term-context matrix with zeros
    tc_matrix = np.zeros((len(vocab), len(vocab)))

    for _, line_tokens in line_tuples:
        for i, target_word in enumerate(line_tokens):
            if target_word in vocab_to_id:
                target_id = vocab_to_id[target_word]
                start = max(0, i - context_window_size)
                end = min(len(line_tokens), i + context_window_size + 1)
                context_words = line_tokens[start:i] + line_tokens[i+1:end]

                for context_word in context_words:
                    if context_word in vocab_to_id:
                        context_id = vocab_to_id[context_word]
                        tc_matrix[target_id, context_id] += 1

    return tc_matrix


def create_PPMI_matrix(term_context_matrix):
    # Calculate row and column sums
    row_sums = np.sum(term_context_matrix, axis=1)
    col_sums = np.sum(term_context_matrix, axis=0)
    total_sum = np.sum(term_context_matrix)

    # Calculate PPMI matrix
    ppmi_matrix = np.log(term_context_matrix * total_sum / (row_sums[:, None] * col_sums[None, :]))

    # Replace negative values with zeros (PPMI truncation)
    ppmi_matrix[ppmi_matrix < 0] = 0

    return ppmi_matrix


def create_tf_idf_matrix(term_document_matrix):
    # Calculate the document frequency (number of documents containing each term)
    doc_frequency = np.sum(term_document_matrix > 0, axis=1)

    # Calculate inverse document frequency (IDF)
    idf = np.log(len(term_document_matrix[0]) / (doc_frequency + 1))

    # Calculate TF-IDF matrix
    tf_idf_matrix = term_document_matrix * idf[:, None]

    return tf_idf_matrix


def compute_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    similarity = dot_product / norm_product if norm_product != 0 else 0
    return similarity

def compute_jaccard_similarity(vector1, vector2):
    intersection = np.sum(np.minimum(vector1, vector2))
    union = np.sum(np.maximum(vector1, vector2))
    similarity = intersection / union if union != 0 else 0
    return similarity

def compute_dice_similarity(vector1, vector2):
    intersection = np.sum(np.minimum(vector1, vector2))
    total = np.sum(vector1) + np.sum(vector2)
    similarity = 2 * intersection / total if total != 0 else 0
    return similarity


def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    target_vector = get_column_vector(term_document_matrix, target_play_index)
    similarities = [similarity_fn(target_vector, get_column_vector(term_document_matrix, i)) for i in range(len(document_names))]
    ranks = np.argsort(similarities)[::-1]
    return ranks

def rank_words(target_word_index, matrix, similarity_fn):
    target_vector = get_row_vector(matrix, target_word_index)
    similarities = [similarity_fn(target_vector, get_row_vector(matrix, i)) for i in range(len(vocab))]
    ranks = np.argsort(similarities)[::-1]
    return ranks


if __name__ == '__main__':
  tuples, document_names, vocab = read_in_shakespeare()

  print('Computing term document matrix...')
  td_matrix = create_term_document_matrix(tuples, document_names, vocab)

  print('Computing tf-idf matrix...')
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)

  print('Computing term context matrix...')
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

  print('Computing PPMI matrix...')
  PPMI_matrix = create_PPMI_matrix(tc_matrix)

  random_idx = random.randint(0, len(document_names)-1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
    ranks = rank_plays(random_idx, td_matrix, sim_fn)
    for idx in range(0, 10):
      doc_id = ranks[idx]
      print('%d: %s' % (idx+1, document_names[doc_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))
