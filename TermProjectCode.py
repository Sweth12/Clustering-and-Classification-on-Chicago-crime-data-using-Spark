from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy import linspace
from matplotlib import cm
from datetime import datetime
import sys
import re

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit main_task1.py <train> <test> <outputfile> ")
        exit(-1)
    
    starttime = datetime.now()
    sc = SparkContext(appName="TermProject")
    spark = SparkSession.builder.appName("Chicago_crime_analysis").getOrCreate()

    crimes = spark.read.csv(sys.argv[1],
                       header = True, 
                        inferSchema=True).cache()

    data = crimes.select('Primary Type','Description')

    regexTokenizer = RegexTokenizer(inputCol="Description", outputCol="words", pattern="\\W")

    # stop words
    stopwords = ["http","https","amp","rt","t","c","the"] 

    # remove stop words
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stopwords)
   
    # encode to label indices
    label_stringIdx = StringIndexer(inputCol = "Primary Type", outputCol = "label")

    # Term Frequency and Inverse Document Frequency
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) 

    # Pipeline
    pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])

    pipelineFit = pipeline.fit(data)
    dataset = pipelineFit.transform(data)

    # Split data into training and test datasets
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)

    # Time taken to preprocess the data
    preprocess = datetime.now()
    preprocess_time = preprocess-starttime

    # Build the models
    lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0)
    nb = NaiveBayes(smoothing=1)

    # Train models with Training Data
    lrModel = lr.fit(trainingData) 
    nbModel = nb.fit(trainingData)

    # Time taken to train the data
    training = datetime.now()
    training_time = training-preprocess

    # Testing data
    predictions = lrModel.transform(testData)
    nbpreds = nbModel.transform(testData)

    # Time taken to test data
    test = datetime.now()
    testing_time = test-training
    total_timetaken = test - starttime

    # Evaluation metrics     
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    a1 = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    f2 = evaluator.evaluate(nbpreds, {evaluator.metricName: "f1"}) 
    a2 = evaluator.evaluate(nbpreds, {evaluator.metricName: "accuracy"})

    print('Logistic Regression F1-Score: ', f1)
    print('Logistic Regression Accuracy: ', a1)
    print('Naive Bayes F1-Score: ', f2)
    print('Naive Bayes Accuracy: ', a2)
    print("Preprocessing",preprocess_time)
    print("Training Time",training_time)
    print("Testing Time",testing_time)
    print("Total time taken",total_timetaken)

    # Saving the f-measure to the file
    sc.parallelize(['Logistic Regression F1-Score: ',f1,'Naive Bayes F1-Score: ',f2, 'Logistic Regression Accuracy: ', a1, 'Naive Bayes Accuracy: ', a2],1).saveAsTextFile(sys.argv[2]+"_F1_measure_and_accuacy")
    sc.parallelize([("Preprocessing",preprocess_time),("Training time",training_time),("Testing time",testing_time),("Total time",total_timetaken)],1).saveAsTextFile(sys.argv[2]+"_Time_taken")

    data1 = crimes.where(crimes.Latitude.isNotNull() & crimes.Longitude.isNotNull() & crimes.ID.isNotNull())

    # Choosing latitude and longitude after removing rows with null values and outliers for better results
    data_frame = data1 \
               .rdd \
               .filter(lambda x: (40.0<float(x[19])<42.0))\
               .filter(lambda x: (-88.0<float(x[20])<-86.0))\
               .map(lambda x: (x[0], Vectors.dense(float(x[19]), float(x[20])))).toDF(["ID", "features"])

    (trainingData1, testData1) = data_frame.randomSplit([0.7, 0.3], seed = 100)

    # Number of cluster choosen
    k=6

    kmeans = KMeans().setK(k).setSeed(1)
    kmeans_model = kmeans.fit(trainingData1)

    # Evaluate clustering by computing Within Set Sum of Squared Errors.
    wssse = kmeans_model.computeCost(trainingData1)
    print("Within Set Sum of Squared Errors = " + str(wssse))

    # cluster center reults
    centers = kmeans_model.clusterCenters()
    print("Cluster centers: ")

    for center in centers:
        print(center)
    
    # predict on the test set
    prediction = kmeans_model.transform(testData1)
    prediction.rdd.saveAsTextFile(sys.argv[2]+"_predictions")

    # Evaluate clustering by computing Silhouette value
    evaluator = ClusteringEvaluator()
    silhouette_value = evaluator.evaluate(prediction)
    print("Silhouette with squared euclidean distance = " + str(silhouette_value))

    prediction_center = prediction \
    .groupBy("prediction") \
    .count()\
    .sort("prediction")

    prediction_center.rdd.saveAsTextFile(sys.argv[2]+'_TotalRecordsInEachCluster')

    cm_subsection = linspace(0.0, 1.0, k)  
    color_list = [ cm.jet(x) for x in cm_subsection ]

    plt.figure(figsize=(13,8))

    # Plot to show clustered data predictions
    for i in range (0,k):
        listx=[]
        listy=[]
        cluster_prediction=prediction.select("features","prediction").where(prediction.prediction==i).limit(250)
        for x,y in cluster_prediction.collect():
            listx.append(x[0])
            listy.append(x[1])
        clusterlabel="cluster_"+str(i)
        plt.scatter(listx, listy,c=color_list[i],marker=".",label=clusterlabel)

    # plotting centroids
    center_x=[]
    center_y=[]
    color_c=[]
    cont=0
    for x,y in centers:
        center_x.append(x)
        center_y.append(y)
        color_c.append(color_list[cont])
        cont += 1    
    plt.scatter(center_x, center_y, s=500, facecolors='none', edgecolors='black',marker="o",linewidths=3)
    plt.scatter(center_x, center_y, s=500, c=color_c, marker="x")
    plt.title('2D Chart')
    plt.legend(loc='lower right')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.grid()
    plt.savefig('kmeansfinal.png')

    sc.stop()

    
