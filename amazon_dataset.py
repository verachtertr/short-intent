import json
import pandas as pd
from recpack.datasets import Dataset
from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem, Filter
from typing import List
from tqdm.auto import tqdm

class AmazonDataset(Dataset):
    """Generic handler for Amazon datasets
    All information on the dataset can be found at https://nijianmo.github.io/amazon/index.html.

    Default processing makes sure that:

    - Each review with score 3 or above is considered positive
    - Each remaining user has interacted with at least 5 items
    - Each remaining item has been interacted with by at least 5 users

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default
        will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised?
        Defaults to True
    :type use_default_filters: bool, optional

    """

    USER_IX = 'reviewerID'
    RATING_IX = 'overall'
    TIMESTAMP_IX = 'unixReviewTime'
    ITEM_IX = 'asin'

    DEFAULT_FILENAME = None
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = None
    """URL to fetch the dataset from."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Adressa dataset

        Filters users and items with not enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinRating(3, self.RATING_IX),
            MinItemsPerUser(5, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]


    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        with open(self.file_path, 'r') as f:

            df = pd.DataFrame.from_records([
                {
                    self.USER_IX: x[self.USER_IX],
                    self.ITEM_IX: x[self.ITEM_IX],
                    self.TIMESTAMP_IX: x[self.TIMESTAMP_IX],
                    self.RATING_IX: x[self.RATING_IX],
                }
                for x in [json.loads(line)for line in tqdm(f.readlines(), desc=f"loading {self.file_path}")]
                if self.USER_IX in x and self.ITEM_IX in x and self.TIMESTAMP_IX in x and self.RATING_IX in x
            ])
    
        return df

    def load_dataframe(self):
        return self._load_dataframe()


class AmazonGamesDataset(AmazonDataset):
    """Handles the Amazon Games dataset
    All information on the dataset can be found at https://nijianmo.github.io/amazon/index.html.

    Default processing makes sure that:

    - Each review with score 3 or above is considered positive
    - Each remaining user has interacted with at least 5 items
    - Each remaining item has been interacted with by at least 5 users

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default
        will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised?
        Defaults to True
    :type use_default_filters: bool, optional

    """

    DEFAULT_FILENAME = "Video_Games_5.json"

class AmazonToysAndGamesDataset(AmazonDataset):
    """Handles the Amazon Toys and Games dataset
    All information on the dataset can be found at https://nijianmo.github.io/amazon/index.html.

    Default processing makes sure that:

    - Each review with score 3 or above is considered positive
    - Each remaining user has interacted with at least 5 items
    - Each remaining item has been interacted with by at least 5 users

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default
        will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised?
        Defaults to True
    :type use_default_filters: bool, optional

    """

    DEFAULT_FILENAME = "Toys_and_Games_5.json"
