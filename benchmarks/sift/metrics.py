#!/usr/bin/env python3
#
#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import time
from typing import Optional

import duckdb
import lance
import numpy as np
import pandas as pd


def recall(actual_sorted: np.ndarray, results: np.ndarray):
    """
    Recall-at-k
    """
    len = results.shape[1]
    recall_at_k = np.array([np.sum([1 if id in results[i, :] else 0 for id in row]) * 1.0 / len
                            for i, row in enumerate(actual_sorted)])
    return (recall_at_k.mean(), recall_at_k.std(), recall_at_k)


def l2_argsort(mat, q):
    """
    argsort of l2 distances

    Parameters
    ----------
    mat: ndarray
        shape is (n, d) where n is number of vectors and d is number of dims
    q: ndarray
        shape is d, this is the query vector
    """
    return np.argsort(((mat - q) ** 2).sum(axis=1))


def cosine_argsort(mat, q):
    """
    argsort of cosine distances

    Parameters
    ----------
    mat: ndarray
        shape is (n, d) where n is number of vectors and d is number of dims
    q: ndarray
        shape is d, this is the query vector
    """
    mat /= np.linalg.norm(mat, axis=1)[:, None]
    q /= np.linalg.norm(q)
    scores = np.dot(mat, q)
    return np.argsort(1-scores)
