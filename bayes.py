from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from cleaner import datardd as data

data = data.sample(False, 0.05, seed=666)
words = data.flatMap(lambda x: x['d'].split()).distinct().sortBy(lambda x: x).collect()
sparselen = len(words)
words = {k:v for v, k in enumerate(words)}

import ipdb; ipdb.set_trace()
