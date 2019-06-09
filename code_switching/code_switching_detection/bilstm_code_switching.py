import csv
import json
import os
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import pandas as pd
import torch
import torch.optim as optim

from .bilstm_crf import BiLSTM_CRF

__all__ = [
    'BilstmCodeSwitching',
    'load_data',
    'prepare_data',
]


class BilstmCodeSwitching:

    EMBEDDING_DIM = 3
    HIDDEN_DIM = 4

    OOV_VALUE = "other"

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"

    TAG_TO_IX = {"en": 0, "es": 1, "other": 2, START_TAG: 3, STOP_TAG: 4}

    def __init__(self):
        self.word_to_ix = {}
        self.vocab_size = 0
        self.__model = None
        self.ix_to_tag = {0: "en", 1: "es", 2: "other"}

    def fit(self, filename: str, epochs=20) -> 'BilstmCodeSwitching':
        """Trains model given training data as pandas DataFrame.

        :param filename:
        :param epochs:
        :return: self
        """
        training_data = load_data(filename)
        training_data = prepare_data(training_data)

        self.word_to_ix = get_vocabulary(training_data)
        self.vocab_size = len(self.word_to_ix)

        self.__model = BiLSTM_CRF(self.vocab_size, self.TAG_TO_IX, self.EMBEDDING_DIM, self.HIDDEN_DIM)
        optimizer = optim.SGD(self.__model.parameters(), lr=0.01, weight_decay=1e-4)

        for epoch in range(epochs):
            for sentence, tags in training_data:
                self.__model.zero_grad()

                # Step 1. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = prepare_sequence(sentence, self.word_to_ix)
                targets = torch.tensor([self.TAG_TO_IX[t] for t in tags], dtype=torch.long)

                # Step 2. Run our forward pass.
                loss = self.__model.neg_log_likelihood(sentence_in, targets)

                # Step 3. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, test_data: Iterable[List[str]]) -> List[List[str]]:
        """Predicts labels for each word in sentences.

        :param test_data: iterable of tokenized texts
        :return: Example:

        [['other', 'en', 'other', 'en', 'en', 'other'],
        ['es', 'es', 'en, 'other']]
        """
        predictions = []

        with torch.no_grad():

            for sentence in test_data:
                cleaned_seq = self.__remove_oov(sentence)
                tensor_seq = prepare_sequence(cleaned_seq, self.word_to_ix)
                if tensor_seq.shape[0] > 0:
                    prediction = self.__model(tensor_seq)[1]
                else:
                    prediction = []

                prediction = [self.ix_to_tag[ix] for ix in prediction]

                i = 0
                modified_prediction = []

                for word in sentence:
                    if word in cleaned_seq:
                        modified_prediction.append(prediction[i])
                        i += 1
                    else:
                        modified_prediction.append(self.OOV_VALUE)
                predictions.append(modified_prediction)

        return predictions

    def evaluate(self, test_data_path: str) -> None:
        """Evaluates precision and recall metrics for each class and prints them.

        :param test_data_path:
        """

        data: pd.DataFrame = load_data(test_data_path)
        prepared_data: List[Tuple[List, List]] = prepare_data(data)
        tokenized_texts = [text[0] for text in prepared_data]

        predictions: List[List] = self.predict(tokenized_texts)
        true_values = [text[1] for text in prepared_data]

        flat_predictions = [tag for sequence in predictions for tag in sequence]
        flat_true_values = [tag for sequence in true_values for tag in sequence]

        for lang in self.ix_to_tag.values():

            print(f"Precision for {lang}: {precision(flat_true_values, flat_predictions, lang)}")
            print(f"Recall for {lang}: {recall(flat_true_values, flat_predictions, lang)}\n")

    def save(self, pathname: str):
        """Saves model to file.

        :param pathname: path to save model and vocabulary
        """
        if not os.path.exists(pathname):
            os.makedirs(pathname)

        torch.save(self.__model.state_dict(), Path(pathname) / 'code_switching_model')

        vocab_path = Path(pathname) / 'vocabulary.json'
        json.dump(self.word_to_ix, vocab_path.open('w', encoding='utf-8'), indent=4)

    def load(self, pathname: str) -> 'BilstmCodeSwitching':
        """Loads serialized model. Model must be initialized.

        :param pathname:
        :param model: initialized model
        :return:
        """
        vocab_path = Path(pathname) / 'vocabulary.json'
        self.word_to_ix = json.load(vocab_path.open())
        self.vocab_size = len(self.word_to_ix)
        self.__model = BiLSTM_CRF(self.vocab_size, self.TAG_TO_IX, self.EMBEDDING_DIM, self.HIDDEN_DIM)
        self.__model.load_state_dict(torch.load(Path(pathname) / 'code_switching_model'))
        self.__model.eval()
        return self

    def __remove_oov(self, seq: List[str]) -> List[str]:
        """Removes out-of-vocabulary words from a sequence.

        :param seq: input sequence
        :return: cleaned sequence
        """
        return [w for w in seq if w in self.word_to_ix]


def get_vocabulary(training_data: List[Tuple[Iterable, Iterable]]) -> Dict[str, int]:
    """Generates vocabulary assigning indices to words.

    :param training_data:
    :return:
    """
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def load_data(path_to_train: str) -> pd.DataFrame:
    """Loads data from csv file and returns a DataFrame with string headers.

    :param path_to_train:
    :return: DataFrame
    """
    train_data = pd.read_csv(path_to_train, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
    train_data.columns = ['tweet_id', 'user_id', 'start', 'end', 'token', 'gold_label']
    return train_data


def prepare_data(df: pd.DataFrame) -> List[Tuple[List, List]]:
    """Returns data in proper format for training.

    :param df: DataFrame
    :return: formatted train data
    """
    train_data = []
    for tweet in df.groupby('tweet_id'):
        tokens = tweet[1].token.values
        labels = tweet[1].gold_label.values
        train_data.append((tokens, labels))
    return train_data


def prepare_sequence(seq, to_ix: Dict):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def precision(true_values: Iterable, predictions: Iterable, positive_class: str) -> float:
    """Calculates precision metric for positive_class. Comparing to sklearn precision_score
    this function set values to False if they match neither positive nor negative class.

    :param true_values:
    :param predictions:
    :param positive_class: the label of class considered positive for the metric
    :return: precision score
    """

    true_positive = len([pred for (pred, gold) in zip(predictions, true_values)
                         if (pred == gold and pred == positive_class)])
    positive = len([pred for pred in predictions if pred == positive_class])

    if positive == 0:  # excludes division by zero
        return 0

    return true_positive / positive


def recall(true_values: Iterable, predictions: Iterable, positive_class: str) -> float:
    """Calculates recall metric for positive_class. Comparing to sklearn precision_score
    this function set values to False if they match neither positive nor negative class.

    :param true_values:
    :param predictions:
    :param positive_class: the label of class considered positive for the metric
    :return: recall score
    """

    true_positive = len([pred for (pred, gold) in zip(predictions, true_values)
                         if (pred == gold and pred == positive_class)])

    all_true = len([value for value in true_values if value == positive_class])

    if all_true == 0:  # excludes division by zero
        return 0

    return true_positive / all_true
