// Databricks notebook source
// MAGIC %md
// MAGIC ## Collaborative Filtering
// MAGIC Collaborative filtering is a machine learning technique that predicts ratings awarded to items by users.
// MAGIC 
// MAGIC ### Import the ALS class
// MAGIC In this exercise, you will use the Alternating Least Squares collaborative filtering algorithm to creater a recommender.

// COMMAND ----------

import org.apache.spark.ml.recommendation.ALS

// COMMAND ----------

// MAGIC %md
// MAGIC ### Load Source Data
// MAGIC The source data for the recommender is in two files - one containing numeric IDs for movies and users, along with user ratings; and the other containing details of the movies.

// COMMAND ----------

val movies = spark.read.option("inferSchema","true").option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/movies.csv")
val ratings = spark.read.option("inferSchema","true").option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/ratings.csv")
ratings.join(movies, "movieId").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Prepare the Data
// MAGIC To prepare the data, split it into a training set and a test set.

// COMMAND ----------

val data = ratings.select("userId", "movieId", "rating")
val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0).withColumnRenamed("rating", "label")
val test = splits(1).withColumnRenamed("rating", "trueLabel")
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Build the Recommender
// MAGIC The ALS class is an estimator, so you can use its **fit** method to traing a model, or you can include it in a pipeline. Rather than specifying a feature vector and as label, the ALS algorithm requries a numeric user ID, item ID, and rating.

// COMMAND ----------

val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("label")
val model = als.fit(train)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test the Recommender
// MAGIC Now that you've trained the recommender, you can see how accurately it predicts known ratings in the test set.

// COMMAND ----------

val prediction = model.transform(test)
prediction.join(movies, "movieId").select("userId", "title", "prediction", "trueLabel").show(100, truncate=false)

// COMMAND ----------

// MAGIC %md
// MAGIC The data used in this exercise describes 5-star rating activity from [MovieLens](http://movielens.org), a movie recommendation service. It was created by GroupLens, a research group in the Department of Computer Science and Engineering at the University of Minnesota, and is used here with permission.
// MAGIC 
// MAGIC This dataset and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.
// MAGIC 
// MAGIC For more information, see F. Maxwell Harper and Joseph A. Konstan. 2015. [The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015)](http://dx.doi.org/10.1145/2827872)
