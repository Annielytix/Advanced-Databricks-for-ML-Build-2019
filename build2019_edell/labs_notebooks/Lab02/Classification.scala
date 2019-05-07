// Databricks notebook source
// MAGIC %md
// MAGIC ## Creating a Classification Model
// MAGIC 
// MAGIC In this exercise, you will implement a classification model that uses features of a flight to predict whether or not it will be late.
// MAGIC 
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC 
// MAGIC First, import the libraries you will need:

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Encoders

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}

// COMMAND ----------

// MAGIC %md ### Load Source Data
// MAGIC The data for this exercise is provided as a CSV file containing details of flights that has already been cleaned up for modeling. The data includes specific characteristics (or *features*) for each flight, as well as a *label* column indicating whether or not the flight was late (a flight with an arrival delay of more than 25 minutes is considered *late*).
// MAGIC 
// MAGIC You will load this data into a dataframe and display it.

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

// MAGIC %md ### Prepare the Training Data
// MAGIC To train the classification model, you need a training data set that includes a vector of numeric features, and a label column. In this exercise, you will use the **StringIndexer** class to generate a numeric category for each discrete **Carrier** string value, and then use the **VectorAssembler** class to transform the numeric features that would be available for a flight that hasn't yet arrived into a vector, and then rename the **Late** column to **label** as this is what we're going to try to predict.
// MAGIC 
// MAGIC *Note: This is a deliberately simple example. In reality you'd likely perform mulitple data preparation steps, and later in this course we'll examine how to encapsulate these steps in to a pipeline. For now, we'll just use the numeric features as they are to define the training dataset.*

// COMMAND ----------

// Carrier is a string, and we need our features to be numeric - so we'll generate a numeric index for each distinct carrier string, and transform the dataframe to add that as a column
val carrierIndexer = new StringIndexer().setInputCol("Carrier").setOutputCol("CarrierIdx")
val numTrain = carrierIndexer.fit(train).transform(train)

// Now we'll assemble a vector of all the numeric feature columns (other than ArrDelay, which we wouldn't have for enroute flights)
val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "CarrierIdx", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
val training = assembler.transform(numTrain).select($"features", $"Late".alias("label"))
training.show()

// COMMAND ----------

// MAGIC %md ### Train a Classification Model
// MAGIC Next, you need to train a classification model using the training data. To do this, create an instance of the classification algorithm you want to use and use its **fit** method to train a model based on the training dataframe. In this exercise, you will use a *Logistic Regression* classification algorithm - but you can use the same technique for any of the classification algorithms supported in the spark.ml API.

// COMMAND ----------

val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
val model = lr.fit(training)
println ("Model trained!")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Prepare the Testing Data
// MAGIC Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **Late** column to **trueLabel**.

// COMMAND ----------

val numTest = carrierIndexer.fit(test).transform(test)
val testing = assembler.transform(numTest).select($"features", $"Late".alias("trueLabel"))
testing.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. You can use this approach to predict delay status for flights where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted status to the actual status.

// COMMAND ----------

val prediction = model.transform(testing)
val predicted = prediction.select($"features", $"probability", $"prediction".cast("Int"), $"trueLabel")
predicted.show(100, truncate=false)

// COMMAND ----------

// MAGIC %md
// MAGIC Looking at the result, the **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data. It looks like there are a mix of correct and incorrect predictions - later in this course you'll learn how to measure the accuracy of a model.
