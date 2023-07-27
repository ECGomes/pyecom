# Provides a helper class for encoding and decoding solutions

import numpy as np


class EncodingConcat:

    def __init__(self, raw: dict):

        self.raw = raw
        self.encoded = None
        self.decoded = None

    @staticmethod
    def _encode(raw: dict):
        # Concatenate the flattened input
        encoded = np.concatenate([value.ravel() for value in raw.values()])

        return encoded

    @staticmethod
    def _decode(encoded: np.ndarray, raw: dict):
        # Decode the flattened input

        decoded = {}
        var_idx = [value.ravel().shape[0] for value in raw.values()]

        current_idx = 0
        names = list(raw.keys())

        for idx, (name, value) in enumerate(zip(names, raw.values())):
            result_idx = current_idx + var_idx[idx]
            decoded[name] = np.reshape(encoded[current_idx:current_idx + var_idx[idx]], value.shape)

            current_idx = result_idx

        return decoded

    def encode(self):
        self.encoded = self._encode(self.raw)
        return

    def decode(self):
        self.decoded = self._decode(self.encoded, self.raw)
        return
