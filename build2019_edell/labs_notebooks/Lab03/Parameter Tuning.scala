// Databricks notebook source
// MAGIC %md
// MAGIC ## Tuning Model Parameters
// MAGIC 
// MAGIC In this exercise, you will optimise the parameters for a classification model.
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
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// Load the source data
case class flight(DayofMonth:Int, DayOfWeek:Int, Carrier:String, OriginAirportID:Int, DestAirportID:Int, DepDelay:Int, ArrDelay:Int, Late:Int)
val flightSchema = Encoders.product[flight].schema
var data = spark.read.schema(flightSchema).option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/flights.csv")
data = data.select($"DayofMonth", $"DayOfWeek", $"Carrier", $"OriginAirportID", $"DestAirportID", $"DepDelay", $"Late".alias("label"))

// Split the data
val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Define the Pipeline
// MAGIC Now define a pipeline that creates a feature vector and trains a classification model

// COMMAND ----------

// Define the pipeline
val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(assembler, lr))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Tune Parameters
// MAGIC You can tune parameters to find the best model for your data. A simple way to do this is to use  **TrainValidationSplit** to evaluate each combination of parameters defined in a **ParameterGrid** against a subset of the training data in order to find the best performing parameters.

// COMMAND ----------

val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01, 0.001)).addGrid(lr.maxIter, Array(10, 5, 2)).build()
val tvs = new TrainValidationSplit().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)

val model = tvs.fit(train)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test the Model
// MAGIC Now you're ready to apply the model to the test data.

// COMMAND ----------

val prediction = model.transform(test)
val predicted = prediction.select("features", "prediction", "probability", "label")
predicted.show(100)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Compute Confusion Matrix Metrics
// MAGIC Now you can examine the confusion matrix metrics to judge the performance of the model.

// COMMAND ----------

val tp = predicted.filter("prediction == 1 AND label == 1").count().toFloat
val fp = predicted.filter("prediction == 1 AND label == 0").count().toFloat
val tn = predicted.filter("prediction == 0 AND label == 0").count().toFloat
val fn = predicted.filter("prediction == 0 AND label == 1").count().toFloat
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
// MAGIC ### Review the Area Under ROC
// MAGIC You can also assess the accuracy of the model by reviewing the area under ROC metric.

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction").setMetricName("areaUnderROC")
val aur = evaluator.evaluate(prediction)
println("AUR = " + (aur))
