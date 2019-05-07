// Databricks notebook source
// MAGIC %md
// MAGIC ## Evaluating a Classification Model
// MAGIC 
// MAGIC In this exercise, you will create a pipeline for a classification model, and then apply commonly used metrics to evaluate the resulting classifier.
// MAGIC 
// MAGIC ### Prepare the Data
// MAGIC 
// MAGIC First, import the libraries you will need and prepare the training and test data:

// COMMAND ----------

// Import Spark SQL and Spark ML libraries
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Encoders

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, MinMaxScaler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

// Load the source data
case class flight(DayofMonth:Int, DayOfWeek:Int, Carrier:String, OriginAirportID:Int, DestAirportID:Int, DepDelay:Int, ArrDelay:Int, Late:Int)
val flightSchema = Encoders.product[flight].schema
var data = spark.read.schema(flightSchema).option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/flights.csv")

// Split the data
val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Define the Pipeline and Train the Model
// MAGIC Now define a pipeline that creates a feature vector and trains a classification model

// COMMAND ----------

// Define the pipeline
val monthdayIndexer = new StringIndexer().setInputCol("DayofMonth").setOutputCol("DayofMonthIdx")
val weekdayIndexer = new StringIndexer().setInputCol("DayOfWeek").setOutputCol("DayOfWeekIdx")
val carrierIndexer = new StringIndexer().setInputCol("Carrier").setOutputCol("CarrierIdx")
val originIndexer = new StringIndexer().setInputCol("OriginAirportID").setOutputCol("OriginAirportIdx")
val destIndexer = new StringIndexer().setInputCol("DestAirportID").setOutputCol("DestAirportIdx")
val numVect = new VectorAssembler().setInputCols(Array("DepDelay")).setOutputCol("numFeatures")
val minMax = new MinMaxScaler().setInputCol(numVect.getOutputCol).setOutputCol("normNums")
val featVect = new VectorAssembler().setInputCols(Array("DayofMonthIdx", "DayOfWeekIdx", "CarrierIdx", "OriginAirportIdx","DestAirportIdx", "normNums")).setOutputCol("features")
val lr = new LogisticRegression().setLabelCol("Late").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(monthdayIndexer, weekdayIndexer, carrierIndexer, originIndexer, destIndexer, numVect, minMax, featVect, lr))

// Train the model
val model = pipeline.fit(train)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test the Model
// MAGIC Now you're ready to apply the model to the test data.

// COMMAND ----------

val prediction = model.transform(test)
val predicted = prediction.select($"features", $"prediction".cast("Int"), $"Late".alias("trueLabel"))
predicted.show(100, truncate=false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Compute Confusion Matrix Metrics
// MAGIC Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
// MAGIC - True Positives
// MAGIC - True Negatives
// MAGIC - False Positives
// MAGIC - False Negatives
// MAGIC 
// MAGIC From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

// COMMAND ----------

val tp = predicted.filter("prediction == 1 AND truelabel == 1").count().toFloat
val fp = predicted.filter("prediction == 1 AND truelabel == 0").count().toFloat
val tn = predicted.filter("prediction == 0 AND truelabel == 0").count().toFloat
val fn = predicted.filter("prediction == 0 AND truelabel == 1").count().toFloat
val metrics = spark.createDataFrame(Seq(
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn)))).toDF("metric", "value")
metrics.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### View the Raw Prediction and Probability
// MAGIC The prediction is based on a raw prediction score that describes a labelled point in a logistic function. This raw prediction is then converted to a predicted label of 0 or 1 based on a probability vector that indicates the confidence for each possible label value (in this case, 0 and 1). The value with the highest confidence is selected as the prediction.

// COMMAND ----------

prediction.select($"rawPrediction", $"probability", $"prediction".cast("Int"), $"Late".alias("trueLabel")).show(100, truncate=false)

// COMMAND ----------

// MAGIC %md
// MAGIC Note that the results include rows where the probability for 0 (the first value in the **probability** vector) is only slightly higher than the probability for 1 (the second value in the **probability** vector). The default *discrimination threshold* (the boundary that decides whether a probability is predicted as a 1 or a 0) is set to 0.5; so the prediction with the highest probability is always used, no matter how close to the threshold.

// COMMAND ----------

// MAGIC %md ### Review the Area Under ROC
// MAGIC Another way to assess the performance of a classification model is to measure the area under a *received operator characteristic (ROC) curve* for the model. The **spark.ml** library includes a **BinaryClassificationEvaluator** class that you can use to compute this. A ROC curve plots the True Positive and False Positive rates for varying threshold values (the probability value over which a class label is predicted). The area under this curve gives an overall indication of the models accuracy as a value between 0 and 1. A value under 0.5 means that a binary classification model (which predicts one of two possible labels) is no better at predicting the right class than a random 50/50 guess.

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator().setLabelCol("Late").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
val auc = evaluator.evaluate(prediction)
println("AUC = " + (auc))
