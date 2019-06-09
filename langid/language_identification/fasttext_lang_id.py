import os
from pathlib import Path
from typing import Iterable
from typing import List

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
    __OUTPUT_TRAINED_MODEL = __DIRNAME / 'models/dialect_clf_model'

    def __init__(self, path_to_model: str):
        self.__model = FastText(path_to_model)

    def fit(self, filename: str, epoch: int = 100) -> 'FasttextLangId':
        """Trains model given filename - path to train data file.

        :param filename: path to train data file
        :param epoch: number of train iterations
        :return: trained model
        """

        self.__model.supervised(input=filename, output=self.__OUTPUT_TRAINED_MODEL, epoch=epoch, lr=0.7)
        return self

    def predict(self, texts: Iterable[str]) -> List[str]:
        """Assigns a class value for each text in texts.

        :param texts: input texts
        :return: list of class values - language codes
        """
        pass

    def predict_file(self, filename: str) -> List[str]:
        """Assigns a class value for each text in texts from file.

        :param filename: file with texts for input
        :return: list of class values - language codes
        """
        pass

    def evaluate(self, filename: str) -> None:
        """Evaluates precision and recall metrics for each class.

        :param filename: file with test set of texts
        """
        pass

    def save(self, filename: str) -> None:
        """Saves model to file.

        :param filename: file to save to
        """
        pass

    @classmethod
    def load(cls, filename: str) -> 'FasttextLangId':
        """Loads model saved with 'save' method.

        :param filename: file to load from
        :return: model -s class instance
        """
        return model
