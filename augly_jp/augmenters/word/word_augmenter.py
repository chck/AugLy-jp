from nlpaug_jp import Augmenter, Data
from nlpaug_jp.utils import Action


class WordAugmenter(Augmenter):
    def __init__(self, action: Action):
        self.action = action
