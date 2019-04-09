from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from cleaner import datardd as data

words = data.flatMap(lambda x: x['d'].split()).distinct().sortBy(lambda x: x).collect()
sparselen = len(words)
words = {k:v for v, k in enumerate(words)}
lbp = data.flatMap(lambda x: [((x['id'], x['cscat'], words[w]), 1) for w in x['d'].split()] )\
        .reduceByKey(lambda x, y: x + y)\
        .map(lambda x: ((x[0][0], x[0][1]), {x[0][2]: x[1]}))\
        .reduceByKey(lambda x, y: {**x, **y})\
        .map(lambda x: LabeledPoint(x[0][1], SparseVector(sparselen, x[1])) )

training, test = lbp.randomSplit([0.6, 0.4])
model = NaiveBayes.train(training, 1.0)
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))

accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))

import ipdb; ipdb.set_trace()
