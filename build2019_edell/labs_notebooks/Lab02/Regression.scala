// Databricks notebook source
// MAGIC %md
// MAGIC ## Creating a Regression Model
// MAGIC 
// MAGIC In this exercise, you will implement a regression model that uses features of a flight to predict how late or early it will arrive.
// MAGIC 
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC 
// MAGIC First, import the libraries you will need:

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.Encoders

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.{StringIndexer,VectorAssembler}

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

// MAGIC %md ### Prepare the Training Data
// MAGIC To train the regression model, you need a training data set that includes a vector of numeric features, and a label column. In this exercise, you will use the **StringIndexer** class to generate a numeric category for each discrete **Carrier** string value, and then use the **VectorAssembler** class to transform the numeric features that would be available for a flight that hasn't yet arrived into a vector, and then rename the **ArrDelay** column to **label** as this is what we're going to try to predict.
// MAGIC 
// MAGIC *Note: This is a deliberately simple example. In reality you'd likely perform mulitple data preparation steps, and later in this course we'll examine how to encapsulate these steps in to a pipeline. For now, we'll just use the numeric features as they are to dewfine the traaining dataset.*

// COMMAND ----------

// Carrier is a string, and we need our features to be numeric - so we'll generate a numeric index for each distinct carrier string, and transform the dataframe to add that as a column
val carrierIndexer = new StringIndexer().setInputCol("Carrier").setOutputCol("CarrierIdx")
val numTrain = carrierIndexer.fit(train).transform(train)

// Now we'll assemble a vector of all the numeric feature columns (other than ArrDelay, which we wouldn't have for enroute flights)
val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "CarrierIdx", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
val training = assembler.transform(numTrain).select($"features", $"ArrDelay".cast("Int").alias("label"))
training.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Train a Regression Model
// MAGIC Next, you need to train a regression model using the training data. To do this, create an instance of the regression algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this exercise, you will use a *Linear Regression* algorithm - though you can use the same technique for any of the regression algorithms supported in the spark.ml API.

// COMMAND ----------

val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
val model = lr.fit(training)
println("Model Trained!")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Prepare the Testing Data
// MAGIC Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **ArrDelay** column to **trueLabel**.

// COMMAND ----------

// Transform the test data to add the numeric carrier index
val numTest = carrierIndexer.fit(test).transform(test)

// Generate the features vector and label
val testing = assembler.transform(numTest).select($"features", $"ArrDelay".cast("Int").alias("trueLabel"))
testing.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. You can use this approach to predict arrival delay for flights where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted number of minutes late or early to the actual arrival delay.

// COMMAND ----------

val prediction = model.transform(testing)
val predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show()

// COMMAND ----------

// MAGIC %md
// MAGIC Looking at the result, the **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data. It looks like there is some variance between the predictions and the actual values (the individual differences are referred to as *residuals*)- later in this course you'll learn how to measure the accuracy of a model.
