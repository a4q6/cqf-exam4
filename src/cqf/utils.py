import json
from copy import deepcopy
from hashlib import sha256

import dill
import numpy as np
import pandas as pd


def to_hash(obj) -> str:
    if isinstance(obj, set):
        return tuple(sorted([to_hash(e) for e in obj]))
    elif isinstance(obj, (tuple, list)):
        return tuple([to_hash(e) for e in obj])
    elif not isinstance(obj, dict):
        return sha256(repr(obj).encode("utf-8")).hexdigest()

    new_obj = deepcopy(obj)
    for k, v in new_obj.items():
        new_obj[k] = to_hash(v)

    return sha256(json.dumps(new_obj, sort_keys=True).encode("utf-8")).hexdigest()
