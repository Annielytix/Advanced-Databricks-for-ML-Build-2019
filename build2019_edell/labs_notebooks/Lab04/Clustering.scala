// Databricks notebook source
// MAGIC %md
// MAGIC ## Clustering
// MAGIC In this exercise, you will use K-Means clustering to segment customer data into five clusters.
// MAGIC 
// MAGIC ### Import the Libraries
// MAGIC You will use the **KMeans** class to create your model. This will require a vector of features, so you will also use the **VectorAssembler** class.

// COMMAND ----------

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md
// MAGIC ### Load Source Data
// MAGIC The source data for your clusters is in a comma-separated values (CSV) file, and incldues the following features:
// MAGIC - CustomerName: The customer's name
// MAGIC - Age: The customer's age in years
// MAGIC - MaritalStatus: The custtomer's marital status (1=Married, 0 = Unmarried)
// MAGIC - IncomeRange: The top-level for the customer's income range (for example, a value of 25,000 means the customer earns up to 25,000)
// MAGIC - Gender: A numeric value indicating gender (1 = female, 2 = male)
// MAGIC - TotalChildren: The total number of children the customer has
// MAGIC - ChildrenAtHome: The number of children the customer has living at home.
// MAGIC - Education: A numeric value indicating the highest level of education the customer has attained (1=Started High School to 5=Post-Graduate Degree
// MAGIC - Occupation: A numeric value indicating the type of occupation of the customer (0=Unskilled manual work to 5=Professional)
// MAGIC - HomeOwner: A numeric code to indicate home-ownership (1 - home owner, 0 = not a home owner)
// MAGIC - Cars: The number of cars owned by the customer.

// COMMAND ----------

val customers = spark.read.option("inferSchema","true").option("header", "true").csv("wasb://spark@<YOUR_ACCOUNT>.blob.core.windows.net/data/customers.csv")
customers.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Create the K-Means Model
// MAGIC You will use the feaures in the customer data to create a Kn-Means model with a k value of 5. This will be used to generate 5 clusters.

// COMMAND ----------

val assembler = new VectorAssembler().setInputCols(Array("Age", "MaritalStatus", "IncomeRange", "Gender", "TotalChildren", "ChildrenAtHome", "Education", "Occupation", "HomeOwner", "Cars")).setOutputCol("features")
val train = assembler.transform(customers)

val kmeans = new KMeans().setFeaturesCol(assembler.getOutputCol).setPredictionCol("cluster").setK(5).setSeed(0)
val model = kmeans.fit(train)
println("Model Created!")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Get the Cluster Centers
// MAGIC The cluster centers are indicated as vector coordinates.

// COMMAND ----------

println("Cluster Centers: ")
model.clusterCenters.foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Predict Clusters
// MAGIC Now that you have trained the model, you can use it to segemnt the customer data into 5 clusters and show each customer with their allocated cluster.

// COMMAND ----------

val prediction = model.transform(train)
prediction.groupBy("cluster").count().orderBy("cluster").show()

// COMMAND ----------

prediction.select("CustomerName", "cluster").show(50)
