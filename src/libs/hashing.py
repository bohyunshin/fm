import hashlib
import numpy as np


class FeatureHasher:
    """
    Feature hashing implementation for CTR prediction
    """

    def __init__(self, n_features=2**18, signed_hash=True):
        self.n_features = n_features
        self.signed_hash = signed_hash

    def _hash_function(self, key, seed=0):
        """Hash function using MD5"""
        hash_obj = hashlib.md5(f"{key}_{seed}".encode())
        return int(hash_obj.hexdigest(), 16)

    def _get_hash_value(self, feature_name, feature_value):
        """Get hash bucket index for a feature"""
        key = f"{feature_name}:{feature_value}"
        bucket = self._hash_function(key) % self.n_features

        if self.signed_hash:
            # Use second hash for sign to reduce collision impact
            sign = 1 if self._hash_function(key, seed=1) % 2 == 0 else -1
            return bucket, sign
        else:
            return bucket, 1

    def transform_sample(self, sample):
        """Transform a single sample to hashed feature vector"""
        feature_vector = np.zeros(self.n_features)

        for feature_name, feature_value in sample.items():
            if feature_value is not None:
                bucket, sign = self._get_hash_value(feature_name, feature_value)
                feature_vector[bucket] += sign

        return feature_vector

    def transform(self, samples):
        """Transform multiple samples"""
        return np.array([self.transform_sample(sample) for sample in samples])


if __name__ == "__main__":
    row = {
        # 'I1': 0.0,
        # 'I2': 35,
        # 'I3': 0.0,
        # 'I4': 1.0,
        # 'I5': 33737.0,
        # 'I6': 21.0,
        # 'I7': 1.0,
        # 'I8': 2.0,
        # 'I9': 3.0,
        # 'I10': 0.0,
        # 'I11': 1.0,
        # 'I12': 0.0,
        # 'I13': 1.0,
        "C1": "05db9164",
        "C2": "510b40a5",
        "C3": "d03e7c24",
        "C4": "eb1fd928",
        "C5": "25c83c98",
        "C6": 0,
        "C7": "52283d1c",
        "C8": "0b153874",
        "C9": "a73ee510",
        "C10": "015ac893",
        "C11": "e51ddf94",
        "C12": "951fe4a9",
        "C13": "3516f6e6",
        "C14": "07d13a8f",
        "C15": "2ae4121c",
        "C16": "8ec71479",
        "C17": "d4bb7bd8",
        "C18": "70d0f5f9",
        "C19": 0,
        "C20": 0,
        "C21": "0e63fca0",
        "C22": 0,
        "C23": "32c7478e",
        "C24": "0e8fe315",
        "C25": 0,
        "C26": 0,
    }

    from sklearn.feature_extraction import FeatureHasher

    hasher = FeatureHasher(n_features=2**16, input_type="dict")  # 65k features
    X = hasher.transform([row])
    print(X.shape)
    print(X.sum())
    print(X)
