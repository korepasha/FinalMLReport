import org.apache.spark.mllib.linalg.{Vectors, Vector, Matrices}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import sqlContext.implicits._
import org.apache.spark.sql.Row
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import Array._
import org.apache.spark.mllib.feature.{ElementwiseProduct, StandardScaler}

val dataFile = "/home/natalia/ML/ICML/fer2013/fer2013.csv"
    val text = sc.textFile("/home/natalia/ML/ICML/fer2013/fer2013.csv").filter(line => !(line.startsWith("emotion")))
    val triplets = text.map(_.split(",")).map(x => (x(0).toInt, x(1).split(' ').map(255.0 - _.toDouble), x(2)))
    triplets.persist(DISK_ONLY)
    val trainSet = triplets.filter({case (res, img, st) => (st == "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    val testSet = triplets.filter({case (res, img, st) => (st != "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    trainSet.persist(DISK_ONLY)
    testSet.persist(DISK_ONLY)
/**
* val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainSet.map(x => x.features))
* val trn1 = trainSet.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)
* val tst1 = testSet.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)
*/

val gaussian = new MultivariateGaussian(Vectors.dense(0.0, 0.0), Matrices.dense(2, 2, Array(1.0, 0.0, 0.0, 1.0)))
val div = 47.0*0.25
val del = 2.0
val wmat = Array.tabulate(48, 48)((a,b) => gaussian.pdf(Vectors.dense(a.toDouble/div-del, b.toDouble/div-del)))
val weights = Vectors.dense(wmat.flatten)
val transformer = new ElementwiseProduct(weights)
val trn = trainSet.map(p => LabeledPoint(p.label, transformer.transform(p.features)))
val tst = testSet.map(p => LabeledPoint(p.label, transformer.transform(p.features)))

val numIter = 500
val regParam = 0.1
val intercept = true
val algorithm = new LogisticRegressionWithLBFGS()
algorithm.optimizer.setNumIterations(numIter).setRegParam(regParam)
val lmodel = algorithm.setNumClasses(7).setIntercept(intercept).run(trn)
val predictionAndLabelTrain = trn.map(p => (lmodel.predict(p.features), p.label))
val predictionAndLabelTest = tst.map(p => (lmodel.predict(p.features), p.label)).persist(DISK_ONLY)
val accuracyTrain = 1.0 * predictionAndLabelTrain.filter(x => x._1 == x._2).count() / trn.count()
val accuracyTest = 1.0 * predictionAndLabelTest.filter(x => x._1 == x._2).count() / tst.count
val metrics = new MulticlassMetrics(predictionAndLabelTest)
println("Summary Statistics")
println(s"Precision = ${metrics.precision}")
println(s"Recall = ${metrics.recall}")
println(s"F1 Score = ${metrics.fMeasure}")

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")

