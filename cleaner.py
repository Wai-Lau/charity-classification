import unicodedata
import re
import math
import random
from pyspark.sql import SparkSession
import random
# Dask imports
import dask.bag as db
import dask.dataframe as df
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

stopwords = ""
with open("./data/stopwords.txt", "r") as f:
    stopwords = f.read()
    stopwords = [w for w in stopwords.split("\n") if w]

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("charity-classification")\
        .getOrCreate()
    return spark

spark = init_spark()
newgoing = spark.read.csv("./data/New Ongoing Programs.csv",
                          encoding="ISO-8859-1", header=True, mode="DROPMALFORMED")\
        .withColumnRenamed("BN/Registration Number", "BN/Registration number")
registered = spark.read.csv("./data/List of registered charities as of Aug 2018.csv",
                            encoding="ISO-8859-1", header=True, mode="DROPMALFORMED")
polione = spark.read.csv("./data/Schedule 7_ Political Activities - Description.csv",
                         encoding="ISO-8859-1", header=True, mode="DROPMALFORMED")
politwo = spark.read.csv("./data/Schedule 7_ Political Activities - Outside Canada.csv",
                         encoding="ISO-8859-1", header=True, mode="DROPMALFORMED")
cats = spark.read.csv("./data/ReducedCategories.csv",
                      encoding="ISO-8859-1", header=True, mode="DROPMALFORMED")
catdict = cats.rdd.map(lambda x: x.asDict()).map(lambda x: (x['cat'],x['cscat'])).collectAsMap()

data = newgoing.join(registered,"BN/Registration number",'full')\
        .withColumnRenamed("BN/Registration number","id")\
        .withColumnRenamed("Program Description","desc")\
        .withColumnRenamed("Program Type OP=ongoing program NP=new program NA=not active","active")\
        .withColumnRenamed("Category Code", "cat")\
        .withColumnRenamed("Website Address","site")\
        .select("id","desc","active","cat","site")\
        .sample(0.05, seed=666)

extradesc = polione.join(politwo.withColumnRenamed("Description","desc2"), "BN/Registration Number", 'full')\
        .withColumnRenamed("BN/Registration Number","id")\
        .withColumnRenamed("Description","desc1")\
        .select("id","desc1","desc2")

datardd = data.join(extradesc, "id", "left")\
        .rdd\
        .map(lambda x: x.asDict())\
        .map(lambda x: {
            **x,
            "d": "".join(
                ch if unicodedata.name(ch).startswith(
                    ('LATIN', 'DIGIT', 'SPACE', 'APOSTROPHE')
                ) else " " \
                for ch in " ".join(
                    [x['desc'] if x['desc'] else "",
                     x['desc1'] if x['desc1'] else "",
                     x['desc2'] if x['desc2'] else ""]
                )
            )
        })\
        .map(lambda x: {
            **x,
            'd': " ".join(w.lower() if w.lower() not in stopwords else "" for w in x['d'].split())
        })\
        .map(lambda x: {**x, 'd': " ".join(re.sub(r'.*\d.*', '', w) for w in x['d'].split())})\
        .map(lambda x: {**x, 'd': " ".join(re.sub(r'.*\'.*', '', w) for w in x['d'].split())})\
        .map(lambda x: {**x, 'd': re.sub(r' +',' ',x['d']).strip()})\
        .map(lambda x: {"d":x['d'], "site":x['site'], "active":x["active"], "cat":x["cat"], "id":x["id"]})\
        .filter(lambda x: x['cat'])\
        .filter(lambda x: x['d'])\
        .map(lambda x: {**x, 'cscat': catdict[re.sub(r'^[0]+',"",x['cat'].strip())]})\
        .map(lambda x: {"d":x['d'], "site":x['site'], "active":x["active"], "cscat":x["cscat"], "cat":x['cat'], "id":x["id"]})
