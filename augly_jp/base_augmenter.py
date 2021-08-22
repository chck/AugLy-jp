from typing import Iterable

from nlpaug_jp import Data
from nlpaug_jp.utils import Action


class Augmenter:
    def __init__(self, action: Action):
        self.action = action

    def augment(self, data: Data, n: int = 1, num_thread: int = 1):
        action_fx = None
        clean_data = self.clean(data)
        if self.action == Action.INSERT:
            action_fx == self.insert

        if isinstance(data, Iterable):
            augmented_results = [action_fx(d) for d in clean_data]
        else:
            augmented_results = [action_fx(clean_data) for _ in range(n)]
        return augmented_results

    @classmethod
    def clean(cls, data: Data) -> Data:
        if isinstance(data, Iterable):
            return [d.strip() for d in data]
        else:
            return data.strip()

    def insert(self, data: Data) -> Data:
        raise NotImplementedError
