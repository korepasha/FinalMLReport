import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.feature.PCA
import breeze.linalg.DenseVector
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import math.max
import org.apache.spark.mllib.evaluation.MulticlassMetrics

    val dataFile = "/home/natalia/ML/ICML/fer2013/fer2013.csv"
    val text = sc.textFile("/home/natalia/ML/ICML/fer2013/fer2013.csv").filter(line => !(line.startsWith("emotion")))
    val triplets = text.map(_.split(",")).map(x => (x(0).toInt, x(1).split(' ').map(_.toDouble), x(2)))
    triplets.persist(DISK_ONLY)
    val trainSet = triplets.filter({case (res, img, st) => (st == "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    val testSet = triplets.filter({case (res, img, st) => (st != "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    trainSet.persist(DISK_ONLY)
    testSet.persist(DISK_ONLY)
/**
 * val nbmodel = NaiveBayes.train(trainSet, lambda = 1, modelType = "multinomial")
 * val predictionAndLabelTrain = trainSet.map(p => (nbmodel.predict(p.features), p.label))
 * val predictionAndLabelTest = testSet.map(p => (nbmodel.predict(p.features), p.label))
 * val accuracyTrain = 1.0 * predictionAndLabelTrain.filter(x => x._1 == x._2).count() / trainSet.count()
 * val accuracyTest = 1.0 * predictionAndLabelTest.filter(x => x._1 == x._2).count() / testSet.count
 */
/**
   * val scaler = new StandardScaler(withMean = false, withStd = true).fit(trainSet.map(x => x.features))
   * val projectedTrain = trainSet.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)
   * val projectedTest = testSet.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)
   * val nbmodel = NaiveBayes.train(projectedTrain, lambda = 1, modelType = "multinomial")
   * val predictionAndLabelTrain = projectedTrain.map(p => (nbmodel.predict(p.features), p.label))
   * val predictionAndLabelTest = projectedTest.map(p => (nbmodel.predict(p.features), p.label))
   * val accuracyTrain = 1.0 * predictionAndLabelTrain.filter(x => x._1 == x._2).count() / projectedTrain.count()
   * val accuracyTest = 1.0 * predictionAndLabelTest.filter(x => x._1 == x._2).count() / projectedTest.count
*/

val pca = new PCA(500).fit(trainSet.map(_.features))
val pTrain = trainSet.map(p => p.copy(features = pca.transform(p.features))).persist(DISK_ONLY)
val pTest = testSet.map(p => p.copy(features = pca.transform(p.features))).persist(DISK_ONLY)
val summaryTrain: MultivariateStatisticalSummary = Statistics.colStats(pTrain.map(p => p.features))
val summaryTest: MultivariateStatisticalSummary = Statistics.colStats(pTest.map(p => p.features))
val mTrain = (new DenseVector(summaryTrain.min.toArray)*(-1.0)).max
val mTest = (new DenseVector(summaryTest.min.toArray)*(-1.0)).max
val addV = max(mTrain, mTest)
val trn = pTrain.map(p => LabeledPoint(p.label, Vectors.dense(p.features.toArray.map(_+addV))))
val tst = pTest.map(p => LabeledPoint(p.label, Vectors.dense(p.features.toArray.map(_+addV))))
val scaler2 = new StandardScaler(withMean = false, withStd = true).fit(trn.map(x => x.features))
val trn1 = trn.map(p => LabeledPoint(p.label, scaler2.transform(p.features))).persist(DISK_ONLY)
val tst1 = tst.map(p => LabeledPoint(p.label, scaler2.transform(p.features))).persist(DISK_ONLY)
val nbmodel = NaiveBayes.train(trn1, lambda = 1, modelType = "multinomial")
val predictionAndLabelTrain = trn1.map(p => (nbmodel.predict(p.features), p.label))
val predictionAndLabelTest = tst1.map(p => (nbmodel.predict(p.features), p.label)).persist(DISK_ONLY)
val accuracyTrain = 1.0 * predictionAndLabelTrain.filter(x => x._1 == x._2).count() / trn1.count()
val accuracyTest = 1.0 * predictionAndLabelTest.filter(x => x._1 == x._2).count() / tst1.count
val metrics = new MulticlassMetrics(predictionAndLabelTest)
val precision = metrics.precision
val recall = metrics.recall
val f1Score = metrics.fMeasure
println("Summary Statistics")
println(s"Precision = $precision")
println(s"Recall = $recall")
println(s"F1 Score = $f1Score")

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")




