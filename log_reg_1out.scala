import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.PCA
import math.floor


    val dataFile = "/home/natalia/ML/ICML/fer2013/fer2013.csv"
    val text = sc.textFile("/home/natalia/ML/ICML/fer2013/fer2013.csv").filter(line => !(line.startsWith("emotion")))
    val triplets = text.map(_.split(",")).map(x => (x(0).toInt, x(1).split(' ').map(x => 255.0 - x.toDouble), x(2)))
    triplets.persist(DISK_ONLY)
    val trainSet = triplets.filter({case (res, img, st) => (st == "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    val testSet = triplets.filter({case (res, img, st) => (st != "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    trainSet.persist(DISK_ONLY)
    testSet.persist(DISK_ONLY)

val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainSet.map(x => x.features))
val trn = trainSet.map(p => LabeledPoint(if (p.label == 6) {1} else {0}, scaler.transform(p.features))).persist(DISK_ONLY)
val tst = testSet.map(p => LabeledPoint(if (p.label == 6) {1} else {0}, scaler.transform(p.features))).persist(DISK_ONLY)

val nclass = 2
val numIter = 500
val regParam = 0.1
val intercept = true
val algorithm = new LogisticRegressionWithLBFGS()
algorithm.optimizer.setNumIterations(numIter).setRegParam(regParam)
val lmodel = algorithm.setNumClasses(nclass).setIntercept(intercept).run(trn)
val predictionAndLabelTrain = trn.map(p => (lmodel.predict(p.features), p.label))
val predictionAndLabelTest = tst.map(p => (lmodel.predict(p.features), p.label)).persist(DISK_ONLY)
val accuracyTrain = 1.0 * predictionAndLabelTrain.filter(x => x._1 == x._2).count() / trn.count()
val accuracyTest = 1.0 * predictionAndLabelTest.filter(x => x._1 == x._2).count() / tst.count
val metrics = new MulticlassMetrics(predictionAndLabelTest)
val labels = metrics.labels
labels.foreach { l =>
    println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
    println(s"Recall($l) = " + metrics.recall(l))
}

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")



