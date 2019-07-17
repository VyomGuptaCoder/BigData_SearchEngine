// Databricks notebook source
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.math._

// COMMAND ----------

val inputPlotSummary = sc.textFile("/FileStore/tables/plot_summaries.txt").map(summary => summary.replaceAll("""([\p{Punct}]|\b\p{IsLetter}{1,2}\b)\s*""", " ").toLowerCase.split("""\s+"""))

// COMMAND ----------

val stopWordsSet = sc.textFile("/FileStore/tables/stop_word_list-026c0.csv").flatMap(stopwords => stopwords.split(",")).collect().toSet

// COMMAND ----------

val filterStopWords = inputPlotSummary.map(document => document.map(w => w).filter(x => stopWordsSet.contains(x) == false))

// COMMAND ----------

val movieSeparate =  sc.textFile("/FileStore/tables/plot_summaries.txt").map(_.split("\t"))

// COMMAND ----------

val moviesCount = movieSeparate.map(m => (m(0), 1)).reduceByKey((m1, m2) => m1 + m2).count()

// COMMAND ----------

val searchTerm = sc.textFile("/FileStore/tables/SearchFile.txt").collect().mkString(" ").toLowerCase()

// COMMAND ----------

val searchTermSplit = searchTerm.toLowerCase().split(" ")

// COMMAND ----------

val searchTermCount = searchTermSplit.length

// COMMAND ----------

val moviesNames = sc.textFile("/FileStore/tables/movie_metadata-ab497.tsv")

// COMMAND ----------

if (searchTermCount == 1) {

  val tf = filterStopWords.map(a => (a(0), a.count(_.contains(searchTerm)).toDouble / a.size)).filter(b => b._2 != 0.0)
  //calculated here term frequency
  
  val df = sc.textFile("/FileStore/tables/plot_summaries.txt").flatMap(a => a.split("\n").filter(b => b.contains(searchTerm))).map(a => ("b", 1)).reduceByKey(_ + _).collect()(0)._2
  //Calculated document frequency
  
  val idf = 1 + math.log(moviesCount / df)
  //Calculated idf (inverse document frequency)
  
  //Now we shall find tf-idf
  val tf_idf = tf.map(c => (c._1, c._2 * idf))
  
  val sort = moviesNames.map(m => (m.split("\t")(0), m.split("\t")(2))).join(tf_idf).map(c => (c._2._1, c._2._2)).sortBy(-_._2).map(_._1).take(10)
    
  val output = sc.parallelize(sort)
  output.collect()
}

// COMMAND ----------

//We handled in command 11 the scenario if only 1 search term is there, following segment handles multiple search terms
if (searchTermCount > 1){
  //tf
  def t_freq(term: String)= filterStopWords.map(d => (d(0), d.count(_.contains(term)).toDouble / d.size)).filter(d => d._2!=0.0)
  val tf = searchTermSplit.map(term => t_freq(term).collect().toMap)
  //df
  def d_freq(term: String) = sc.textFile("/FileStore/tables/plot_summaries.txt").flatMap(m => m.split("\n").filter(sum => sum.contains(term))).map(m => ("summary", 1)).reduceByKey(_ + _).collect()(0)._2
  val df = searchTermSplit.map(doc => d_freq(doc))
  
  //idf
  val idf = df.map(x => (1+ math.log(moviesCount/x)))
  
  //tf-idf
  def tf_idf(a: Int) = tf(a).map(b=>(b._1,b._2*idf(a))).toMap
  val tf_idf1 = tf.zipWithIndex.map{ case (a, b) =>tf_idf(b) }
  val tf_find =  searchTermSplit.map(a => searchTermSplit.count(_.contains(a)).toDouble/searchTermCount)
  val tf_idf_find= tf_find.zipWithIndex.map{case (a, b) => a * idf(b)}
  val calc = math.sqrt(tf_idf_find.reduce((x, y) => x * x + y * y))
  val distinct = tf_idf1.flatMap(x => x.map(y => y._1)).toList.distinct.toArray
  
  //Cosine similarity Implementation
  def value(term:String)= searchTermSplit.zipWithIndex.map{case (a, b) => (tf_idf1(b).get(term).getOrElse(0.0).asInstanceOf[Double]).toDouble }.reduce((x,y)=>x*x+y*y)
  val document = distinct.map(s =>  (s, math.sqrt(value(s)))).toMap
  def dotProduct(value:String)= searchTermSplit.zipWithIndex.map{case (x, y) => (tf_idf_find(y) * tf_idf1(y).get(value).getOrElse(0.0).asInstanceOf[Double]).toDouble }.reduce((x,y)=>x+y)
  val dotProductVal = distinct.map(s =>  (s, dotProduct(s))).toMap
  val cosCalc= distinct.map( s => (s, dotProductVal.get(s).getOrElse(0.0).asInstanceOf[Double] / (document.get(s).getOrElse(0.0).asInstanceOf[Double] * calc)) )
  val cos = sc.parallelize(cosCalc)
  
  //Output
  val sort1 = moviesNames.map(m => (m.split("\t")(0), m.split("\t")(2))).join(cos).map(s =>(s._2._1, s._2._2)).sortBy(_._2).map(_._1).take(10)
  val output1 = sc.parallelize(sort1)
  output1.collect()
}

