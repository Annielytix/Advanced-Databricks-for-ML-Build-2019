// Databricks notebook source
// MAGIC %md
// MAGIC ## Creating a Pipeline
// MAGIC 
// MAGIC In this exercise, you will implement a pipeline that includes multiple stages of *transformers* and *estimators* to prepare features and train a classification model. The resulting trained *PipelineModel* can then be used as a transformer to predict whether or not a flight will be late.
// MAGIC 
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC 
// MAGIC First, import the libraries you will need:

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Encoders

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, MinMaxScaler}
import org.apache.spark.ml.classification.LogisticRegression

// COMMAND ----------

// MAGIC %md
// MAGIC ### Load Source Data
// MAGIC The data for this exercise is provided as a CSV file containing details of flights. The data includes specific characteristics (or *features*) for each flight, as well as a column indicating how many minutes late or early the flight arrived.
// MAGIC 
// MAGIC You will load this data into a DataFrame and display it.

// COMMAND ----------

case class flight(DayofMonth:Int, DayOfWeek:Int, Carrier:String, OriginAirportID:Int, DestAirportID:Int, DepDelay:Int, ArrDelay:Int, Late:Int)

val flightSchema = Encoders.product[flight].schema

var data = spark.read.schema(flightSchema).option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/flights.csv")
data.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Split the Data
// MAGIC It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.

// COMMAND ----------

val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// MAGIC %md ### Define the Pipeline
// MAGIC A predictive model often requires multiple stages of feature preparation. For example, it is common when using some algorithms to distingish between continuous features (which have a calculable numeric value) and categorical features (which are numeric representations of discrete categories). It is also common to *normalize* continuous numeric features to use a common scale - for example, by scaling all numbers to a proportional decimal value between 0 and 1 (strictly speaking, it only really makes sense to do this when you have multiple numeric columns - normalizing them all to similar scales prevents a feature with particularly large values from dominating the training of the model - in this case, we only have one non-categorical numeric feature; but I've included this so you can see how it's done!).
// MAGIC 
// MAGIC A pipeline consists of a a series of *transformer* and *estimator* stages that typically prepare a dataframe for
// MAGIC modeling and then train a predictive model. In this case, you will create a pipeline with seven stages:
// MAGIC - A **StringIndexer** estimator for each categorical variable to generate numeric indexes for categorical features
// MAGIC - A **VectorAssembler** that creates a vector of continuous numeric features
// MAGIC - A **MinMaxScaler** that normalizes vector of numeric features
// MAGIC - A **VectorAssembler** that creates a vector of categorical and continuous features
// MAGIC - A **LogisticRegression** algorithm that trains a classification model.

// COMMAND ----------

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

// COMMAND ----------

// MAGIC %md
// MAGIC ### Run the Pipeline as an Estimator
// MAGIC The pipeline itself is an estimator, and so it has a **fit** method that you can call to run the pipeline on a specified DataFrame. In this case, you will run the pipeline on the training data to train a model.

// COMMAND ----------

val model = pipeline.fit(train)
println("Pipeline complete!")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test the Pipeline Model
// MAGIC The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the **test** DataFrame using the pipeline to generate label predictions.

// COMMAND ----------

val prediction = model.transform(test)
val predicted = prediction.select($"features", $"prediction".cast("Int"), $"Late".alias("trueLabel"))
predicted.show(100, truncate=false)

// COMMAND ----------

// MAGIC %md
// MAGIC The resulting DataFrame is produced by applying all of the transformations in the pipline to the test data. The **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data.
