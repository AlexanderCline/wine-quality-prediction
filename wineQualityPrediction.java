package wineQualityPrediction;

import java.io.IOException;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class wineQualityPrediction {

	public static void main(String[] args) throws IOException {
		
		//HDFS instance
		String hdfs = args[0];
		
		//Initialize the Spark session
		SparkSession spark = SparkSession
			    .builder()
			    .appName("Wine Quality Predictor - Random Forest Classifier")
			    .getOrCreate();
		
		//Load the trained model
		System.out.println("Loading trained model...");
		String model = hdfs + "/trained.model";
		PipelineModel loadedModel = PipelineModel.load(model);
		
		//Load the prediction test dataset
		System.out.println("Loading prediction test data...");
		String testDataset = hdfs + "/" + args[1] ;
		Dataset<Row> test = spark.read().format("csv")
				.option("sep", ";")
				.option("inferSchema", true)
				.option("header", true)
				.load(testDataset);
		
		//Create an evaluator for the model
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setPredictionCol("prediction")
				.setLabelCol("\"\"\"\"quality\"\"\"\"\"")
				.setMetricName("f1");
		
		//Evaluate the model using the test data
		System.out.println("Predicting...");
		Dataset<Row> prediction = loadedModel.transform(test);
		double f1Score = evaluator.evaluate(prediction);

		prediction.select("\"\"\"\"quality\"\"\"\"\"", "prediction").show();
		
		spark.stop();
		
		System.out.println("F1 Score of the trained model ----------------------------> " + f1Score);
	}
}
