package com.training.wineQuality;

import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class wineQualityTraining {

	public static void main(String[] args) throws IOException {
		
		//HDFS instance
		String hdfs = args[0];
		
		//Flag if we want to validate the model against our ValidationDataset
		//Ask before we begin the Spark application
		boolean validate = runValidation();
		
		//Create the Spark session
		SparkSession spark = SparkSession
			    .builder()
			    .appName("Wine Quality Training - Random Forest Classifier")
			    .getOrCreate();
		
		System.out.println("Loading the training data...");
		
		//Load training data
		Dataset<Row> input = spark.read().format("csv")
				.option("sep", ";")
				.option("inferSchema", true)
				.option("header", true)
				.load(hdfs + "/TrainingDataset.csv");
		
		//Create features column, but don't include """"quality"""" in it
		String[] featureColumns = input.columns();
		featureColumns = Arrays.copyOf(featureColumns, featureColumns.length-1);
		
		//Helps create our feature column
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(featureColumns)
				.setOutputCol("features");
		
		System.out.println("Begin training the model...");
		
		//RandomForestClassfier is our model
		RandomForestClassifier rf = new RandomForestClassifier()
				.setFeaturesCol("features")
				.setLabelCol("\"\"\"\"quality\"\"\"\"\"")
				.setNumTrees(55)
				.setMaxBins(128)
				.setMaxDepth(20)
				.setMinInstancesPerNode(5);
		
		//Create a pipeline for the model
		Pipeline rfPipeline = new Pipeline()
				.setStages(new PipelineStage[] {assembler, rf});
		PipelineModel rfPipelineModel = rfPipeline.fit(input);
		
		//If you want to validate the model
		if(validate)
			validateModel(rfPipelineModel, spark, hdfs);
		
		//Save the Pipeline Model so it can be used by the prediction application
		System.out.println("Model has finished training...");
		String outputModel = hdfs + "/trained.model";
		System.out.println("Saving model to: " + outputModel);
		rfPipelineModel.write().overwrite().save(outputModel);

		spark.stop();

	}

	/***
	 * 
	 * runValidation method
	 * 
	 * Asks if the user would like to validate the data before saving the model
	 * @return boolean
	 */
	public static boolean runValidation() {
		
		boolean validate = false;
		boolean answered = true;
		String read;
		Scanner scanner = new Scanner(System.in);
		do{
			System.out.println("Do you want to validate the model using the validation data? (y/n)");
			read = scanner.nextLine();
			if(read.equals("y") || read.equals("n")) {
				answered = false;
				if(read.equals("y"))
					validate = true;
				else
					validate = false;
			}
			else
				System.out.println("ERR: Expecting y or n");
		}while(answered);
		scanner.close();
		
		return validate;
	}
	
	/***
	 * 
	 * validateModel method
	 * 
	 * Uses the validation data to see the F1 Score of the model is
	 * Used for testing and tuning of the model
	 * @param rfPipelineModel
	 * @param spark
	 */
	public static void validateModel(PipelineModel rfPipelineModel, SparkSession spark, String hdfs) {
		
		//Load validation data
		Dataset<Row> validation = spark.read().format("csv")
				.option("sep", ";")
				.option("inferSchema", true)
				.option("header", true)
				.load(hdfs + "/ValidationDataset.csv");
		
		//Evaluator
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setPredictionCol("prediction")
				.setLabelCol("\"\"\"\"quality\"\"\"\"\"")
				.setMetricName("f1");
		
		//Transform the ValidationDataset on the PipelineModel and check the F1 Score
		Dataset<Row> validate = rfPipelineModel.transform(validation);
		double f1Score = evaluator.evaluate(validate);
		System.out.println("F1 Score: " + f1Score);
	}
}
