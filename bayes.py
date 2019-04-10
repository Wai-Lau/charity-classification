from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark.sql import SparkSession
from cleaner import catdict

# from cleaner import datardd as data
# words = data.flatMap(lambda x: x['d'].split()).distinct().sortBy(lambda x: x).collect()
# sparselen = len(words)
# words = {k:v for v, k in enumerate(words)}
# lbp = data.flatMap(lambda x: [((x['id'], x['cat'], words[w]), 1) for w in x['d'].split()] )\
#         .reduceByKey(lambda x, y: x + y)\
#         .map(lambda x: ((x[0][0], x[0][1]), {x[0][2]: x[1]}))\
#         .reduceByKey(lambda x, y: {**x, **y})\
#         .map(lambda x: LabeledPoint(x[0][1], SparseVector(sparselen, x[1])) )
# lbp.saveAsPickleFile("lbp");

ctx = SparkSession \
    .builder \
    .appName("charity-classification")\
    .getOrCreate().sparkContext
lbp = ctx.pickleFile("./lbp")
lbp = lbp.map(lambda x: LabeledPoint(catdict[str(int(x.label))], x.features))

training, test = lbp.randomSplit([0.7, 0.3], seed=666)

for sample_size in [1]:
    sample = training.sample(False, sample_size, seed=666)
    model = NaiveBayes.train(sample, 1.0)
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
    print(sample_size, "(", sample.count() ,")")
    print('model accuracy {}'.format(accuracy))
    import ipdb; ipdb.set_trace()

