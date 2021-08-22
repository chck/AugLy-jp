from nlpaug_jp import Augmenter, Data, Iterable
from nlpaug_jp.utils import Action


class CharAugmenter(Augmenter):
    def __init__(self, action: Action):
        super().__init__(action=action)

    def clean(cls, data: Data) -> Data:
        if isinstance(data, Iterable):
            return [d.strip() for d in data]
        else:
            return data.strip()
