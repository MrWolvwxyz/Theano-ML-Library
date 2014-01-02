import os
import re
import theano
import string
import time

import theano.tensor as T
from theano.sandbox.linalg.ops import MatrixInverse as matrix_inverse

import numpy as np
import numpy.random as random

from Sparse_Coding import *

sample_input = random.rand( 5, 4 )

model = sparsity_model( sample_input )
model.feature_sign_search()




