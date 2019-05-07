// Databricks notebook source
// MAGIC %md
// MAGIC ## Getting Started with MMLSpark
// MAGIC In this exercise, you will use the Microsoft Machine Learning for Spark (MMLSpark) library to create a classifier.
// MAGIC 
// MAGIC ### Load the Data
// MAGIC First, you'll load the flight delay data from your Azure storage account and create a dataframe with a **Late** column that will be the label your classifier predicts.

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

val csv = spark.read.option("inferSchema","true").option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/flights.csv")
val data = csv.select($"DayofMonth", $"DayOfWeek", $"OriginAirportID", $"DestAirportID", $"DepDelay", ($"ArrDelay" > 15).cast("Int").alias("Late"))
data.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Split the Data for Training and Testing
// MAGIC Now you'll split the data into two sets; one for training a classification model, the other for testing the trained model.

// COMMAND ----------

val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println ("Training Rows:" + train_rows + " Testing Rows:" + test_rows)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Train a Classification Model
// MAGIC The steps so far have been identical to those used to prepare data for training using SparkML. Now we'll use the MMLSpark **TrainClassifier** function to initialize and fit a Logistic Regression model. This function abstracts the various SparkML classes used to do this, implicitly converting the data into the correct format for the algorithm.

// COMMAND ----------

import com.microsoft.ml.spark.TrainClassifier
import org.apache.spark.ml.classification.LogisticRegression
val model = new TrainClassifier().setModel(new LogisticRegression).setLabelCol("Late").setNumFeatures(256).fit(train)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Evaluate the Model
// MAGIC The MMLSpark library also includes classes to calculate the performance metrics of a trained model. The following code calculates metrics for a classifier, and stores them in a table.

// COMMAND ----------

import com.microsoft.ml.spark.ComputeModelStatistics
import com.microsoft.ml.spark.TrainedClassifierModel

val prediction = model.transform(test)
val metrics = new ComputeModelStatistics().transform(prediction)
metrics.createOrReplaceTempView("classMetrics")
metrics.show()

// COMMAND ----------

// MAGIC %md
// MAGIC If the output above is too wide to view clearly, run the following cell to display the results as a scrollable table. The metrics include:
// MAGIC - predicted_class_as_0.0_actual_is_0.0 (true negatives)
// MAGIC - predicted_class_as_0.0_actual_is_1.0 (false negatives)
// MAGIC - predicted_class_as_1.0_actual_is_0.0 (false positives)
// MAGIC - predicted_class_as_1.0_actual_is_1.0 (true positives)
// MAGIC - accuracy (proportion of correct predictions)
// MAGIC - precision (proportion of predicted positives that are actually positive)
// MAGIC - recall (proportion of actual positives correctly predicted by the model)
// MAGIC - AUC (area under the ROC curve indicating true positive rate vs false positive rate for all thresholds)

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM classMetrics

// COMMAND ----------

// MAGIC %md
// MAGIC ### Learn More
// MAGIC This exercise has shown a simple example of using the MMLSpark library. The library really provides its greatest value when building deep learning models with the Microsoft cognitive toolkit (CNTK). To learn more about the MMLSpark library, see https://github.com/Azure/mmlspark.
