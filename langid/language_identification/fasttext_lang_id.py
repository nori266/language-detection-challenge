import os
from pathlib import Path
import pickle
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

from pyfasttext import FastText

__all__ = [
    'FasttextLangId',
]


class FasttextLangId:
    """FasttextLangId is a wrapper for a FastText language identification model.
    It has methods like in scikit-learn models: fit, predict.

    Examples:
    """

    __DIRNAME = Path(os.path.dirname(__file__))
    __OUTPUT_TRAINED_MODEL = __DIRNAME / 'models/langid_variants'
    __NO_PREDICTION_VALUE = 'None'
    __CLASSES = ['pt-pt', 'pt-br']
    __DIALECT_MAP = {'ptpt': 'pt-pt', 'ptbr': 'pt-br'}

    def __init__(self, path_to_model: Optional[str] = None):

        if path_to_model is None:
            self.__model = FastText()
        else:
            self.__model = FastText(path_to_model)

        self.classes = self.__model.labels

    def fit(self, filename: str, output_filename: str, epoch: int = 100, lr=0.7) -> 'FasttextLangId':
        """Trains model given filename - path to train data file.

        :param filename: path to train data file
        :param output_filename:
        :param epoch: number of train iterations
        :param lr:
        :return: trained model
        """

        self.__model.supervised(input=filename, output=output_filename, epoch=epoch, lr=0.7)
        self.classes = self.__model.labels
        return self

    def predict(self, texts: Iterable[str]) -> List[str]:
        """Assigns a class value for each text in texts.

        :param texts: input texts
        :return: list of class values - language codes
        """

        predictions = self.__model.predict(texts)
        predictions = [pred[0] if pred else self.__NO_PREDICTION_VALUE for pred in predictions]

        return [
            self.__DIALECT_MAP[label]
            if label in self.__DIALECT_MAP else label
            for label in predictions
        ]

    def predict_file(self, filename: str) -> List[str]:
        """Assigns a class value for each text in texts from file.

        :param filename: file with texts for input
        :return: list of class values - language codes
        """

        predictions = self.__model.predict_file(filename)
        predictions = [pred[0] if pred else self.__NO_PREDICTION_VALUE for pred in predictions]

        return [
            self.__DIALECT_MAP[label]
            if label in self.__DIALECT_MAP else label
            for label in predictions
        ]

    def evaluate(
            self,
            texts: Union[Iterable[str], str],
            gold_labels: Iterable[str],
            classes: Optional[List[str]] = None
    ) -> None:
        """Evaluates precision and recall metrics for each class and prints them.

        :param texts: iterable of texts or file with test set of texts
        :param gold_labels: ground truth
        :param classes: classes of model to evaluate
        """

        if classes is not None:
            self.classes = classes

        gold_labels = [
            self.__DIALECT_MAP[label]
            if label in self.__DIALECT_MAP else label
            for label in gold_labels
        ]

        if isinstance(texts, Iterable):
            predictions = self.predict(texts)
        else:
            predictions = self.predict_file(texts)

        for class_label in self.classes:
            print(f"Precision for class {class_label}: {precision(gold_labels, predictions, class_label)}")
            print(f"Recall for class {class_label}: {recall(gold_labels, predictions, class_label)}\n")

    def save(self, filename: str) -> None:
        """Saves model to file.

        :param filename: file to save to
        """

        pickle.dump(self, (filename))

    @classmethod
    def load(cls, filename: str) -> 'FasttextLangId':
        """Loads model saved with 'save' method.

        :param filename: file to load from
        :return: model -s class instance
        """
        return pickle.load(filename)


def precision(true_values: Iterable, predictions: Iterable, positive_class: str) -> float:
    """Calculates precision metric for positive_class. Comparing to sklearn precision_score
    this function set values to False if they match neither positive nor negative class.

    :param true_values:
    :param predictions:
    :param positive_class: the label of class considered positive for the metric: must be 'ptpt' or 'ptbr'
    :return: precision score
    """

    true_positive = len([pred for (pred, gold) in zip(predictions, true_values)
                         if (pred == gold and pred == positive_class)])
    positive = len([pred for pred in predictions if pred == positive_class])

    return true_positive / positive


def recall(true_values: Iterable, predictions: Iterable, positive_class: str) -> float:
    """Calculates recall metric for positive_class. Comparing to sklearn precision_score
    this function set values to False if they match neither positive nor negative class.

    :param true_values:
    :param predictions:
    :param positive_class: the label of class considered positive for the metric: must be 'ptpt' or 'ptbr'
    :return: recall score
    """

    true_positive = len([pred for (pred, gold) in zip(predictions, true_values)
                         if (pred == gold and pred == positive_class)])
    all_true = len([value for value in true_values if value == positive_class])

    return true_positive / all_true
