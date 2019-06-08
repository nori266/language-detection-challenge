from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import pandas as pd
import torch


def load_data(path_to_train: str) -> pd.DataFrame:
    """Loads data from csv file and returns a DataFrame with string headers.

    :param path_to_train:
    :return: DataFrame
    """
    train_data = pd.read_csv(path_to_train, sep='\t', header=None)
    train_data.columns = ['tweet_id', 'user_id', 'start', 'end', 'token', 'gold_label']
    return train_data


def prepare_train_data(df: pd.DataFrame) -> List[Tuple[Iterable, Iterable]]:
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


def prepare_sequence(seq, to_ix: Dict):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
