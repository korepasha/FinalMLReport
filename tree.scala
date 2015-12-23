import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.feature.PCA


    val dataFile = "/home/natalia/ML/ICML/fer2013/fer2013.csv"
    val text = sc.textFile("/home/natalia/ML/ICML/fer2013/fer2013.csv").filter(line => !(line.startsWith("emotion")))
    val triplets = text.map(_.split(",")).map(x => (x(0).toInt, x(1).split(' ').map(255.0 - _.toDouble), x(2)))
    triplets.persist(DISK_ONLY)
    val trainSet = triplets.filter({case (res, img, st) => (st == "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    val testSet = triplets.filter({case (res, img, st) => (st != "Training")}).map({case (res, img, st) => LabeledPoint(res, Vectors.dense(img))})
    trainSet.persist(DISK_ONLY)
    testSet.persist(DISK_ONLY)

val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainSet.map(x => x.features))
val trn1 = trainSet.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)
val tst1 = testSet.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)

val pca = new PCA(10).fit(trn1.map(_.features))
val trn = trn1.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)
val tst = tst1.map(p => LabeledPoint(p.label, scaler.transform(p.features))).persist(DISK_ONLY)

val numClasses = 7
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 7
val maxBins = 16

val model = DecisionTree.trainClassifier(trn, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
val predictionAndLabelTrain = trn.map(p => (model.predict(p.features), p.label))
val predictionAndLabelTest = tst.map(p => (model.predict(p.features), p.label)).persist(DISK_ONLY)
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

