
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor,float32
from tensorflow import data as tfdata
from tensorflow import config as tfconfig
from tensorflow import nn
from tensorflow.python.keras import Model, Sequential, Input
from tensorflow.python.keras.layers import GRU, LSTM, Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError

import numpy as np
#from tqdm import tqdm, trange

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Prediction:
    def __init__(self):
        self.date=None

    def create_data(self):
        print("testing")