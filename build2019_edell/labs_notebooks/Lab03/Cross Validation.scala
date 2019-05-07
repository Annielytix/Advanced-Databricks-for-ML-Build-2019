// Databricks notebook source
// MAGIC %md
// MAGIC ## Using Cross Validation
// MAGIC 
// MAGIC In this exercise, you will use cross-validation to optimize parameters for a regression model.
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
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Load the source data
case class flight(DayofMonth:Int, DayOfWeek:Int, Carrier:String, OriginAirportID:Int, DestAirportID:Int, DepDelay:Int, ArrDelay:Int, Late:Int)
val flightSchema = Encoders.product[flight].schema
var data = spark.read.schema(flightSchema).option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/flights.csv")
data = data.select($"DayofMonth", $"DayOfWeek", $"Carrier", $"OriginAirportID", $"DestAirportID", $"DepDelay", $"ArrDelay".alias("label"))

// Split the data
val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Define the Pipeline
// MAGIC Now define a pipeline that creates a feature vector and trains a regression model

// COMMAND ----------

// Define the pipeline
val monthdayIndexer = new StringIndexer().setInputCol("DayofMonth").setOutputCol("DayofMonthIdx")
val weekdayIndexer = new StringIndexer().setInputCol("DayOfWeek").setOutputCol("DayOfWeekIdx")
val carrierIndexer = new StringIndexer().setInputCol("Carrier").setOutputCol("CarrierIdx")
val originIndexer = new StringIndexer().setInputCol("OriginAirportID").setOutputCol("OriginAirportIdx")
val destIndexer = new StringIndexer().setInputCol("DestAirportID").setOutputCol("DestAirportIdx")
val numVect = new VectorAssembler().setInputCols(Array("DepDelay")).setOutputCol("numFeatures")
val minMax = new MinMaxScaler().setInputCol(numVect.getOutputCol).setOutputCol("normNums")
val featVect = new VectorAssembler().setInputCols(Array("DayofMonthIdx", "DayOfWeekIdx", "CarrierIdx", "OriginAirportIdx","DestAirportIdx", "normNums", "DepDelay")).setOutputCol("features")
val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(monthdayIndexer, weekdayIndexer, carrierIndexer, originIndexer, destIndexer, numVect, minMax, featVect, lr))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Tune Parameters
// MAGIC You can tune parameters to find the best model for your data. To do this you can use the  **CrossValidator** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets, in order to find the best performing parameters. Note that this can take a long time to run because every parameter combination is tried multiple times.

// COMMAND ----------

val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.3, 0.01)).addGrid(lr.maxIter, Array(10, 5)).build()
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)

val model = cv.fit(train)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test the Model
// MAGIC Now you're ready to apply the model to the test data.

// COMMAND ----------

val prediction = model.transform(test)
val predicted = prediction.select("features", "prediction", "label")
predicted.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Examine the Predicted and Actual Values
// MAGIC You can plot the predicted values against the actual values to see how accurately the model has predicted. In a perfect model, the resulting scatter plot should form a perfect diagonal line with each predicted value being identical to the actual value - in practice, some variance is to be expected.
// MAGIC Run the cells below to create a temporary table from the **predicted** DataFrame and then retrieve the predicted and actual label values using SQL. You can then display the results as a scatter plot to see if the predicted delays correlate to the actual delays.

// COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT label, prediction FROM regressionPredictions

// COMMAND ----------

// MAGIC %md
// MAGIC ### Retrieve the Root Mean Square Error (RMSE)
// MAGIC There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the prediced and actual values - so in this case, the RMSE indicates the average number of minutes between predicted and actual flight delay values. You can use the **RegressionEvaluator** class to retrieve the RMSE.

// COMMAND ----------

val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(prediction)
println("Root Mean Square Error (RMSE): " + (rmse))
