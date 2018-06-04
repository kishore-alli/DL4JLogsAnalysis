package com.au.westpac.wdp.ai.transform;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.TimeZone;

public class ParseLogFile {
    
    private static Logger log=LoggerFactory.getLogger(ParseLogFile.class);

    public static  void main(String[] args) throws Exception {

        Schema inputDataSchema = new Schema.Builder()
            .addColumnString("ARN")
            .addColumnInteger("ELAPSEDMS")
                .addColumnsString("MESSAGE"  )
                .addColumnsString("PRIORITY")
                .addColumnTime("_time",DateTimeZone.UTC)

            .build();

        log.info("Input data schema details:");
        log.info(inputDataSchema.toString());

        log.info("Other information obtainable from schema:");
        log.info("Number of columns: " + inputDataSchema.numColumns());
        log.info("Column names: " + inputDataSchema.getColumnNames());
        log.info("Column types: " + inputDataSchema.getColumnTypes());

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
/*
                .filter(new ConditionFilter(
                        new StringRegexColumnCondition("MESSAGE", "^$")))
*/
/*
                .filter(new ConditionFilter(
                        new StringRegexColumnCondition("_time", "^.*\\berror\\b.*")))
                .filter(new ConditionFilter(
                        new StringRegexColumnCondition("_time", "^$")))
*/


/*
                .filter(new ConditionFilter(
                        new StringRegexColumnCondition("ELAPSEDMS", "^$")))
*/





                .addConstantIntegerColumn("declined",0)
                .conditionalReplaceValueTransform("declined",new IntWritable(1)
                        ,new StringRegexColumnCondition("MESSAGE",".*\\bDECLINED\\b.*"))
                .conditionalReplaceValueTransform("ELAPSEDMS",new IntWritable(0)
                        ,new StringRegexColumnCondition("ELAPSEDMS","$"))

                .conditionalReplaceValueTransform("PRIORITY",new IntWritable(0)
                         ,new StringRegexColumnCondition("PRIORITY",".*\\bINFO\\b.*"))
                 .conditionalReplaceValueTransform("PRIORITY",new IntWritable(1)
                         ,new StringRegexColumnCondition("PRIORITY",".*\\bERROR\\b.*"))



                .conditionalReplaceValueTransform("MESSAGE",new IntWritable(1)
                        ,new StringRegexColumnCondition("MESSAGE",".*\\bgetLogoDataFromVPlus\\b.*"))

                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(2)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bgetBtplanForWpacBlntrnsfrs\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(3)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\blaunchCardOrFlexiLoanApplication\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(4)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\blaunchWorkflow\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(5)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bpreCISValidation\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(6)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bidentifyCustomers\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(7)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bmaintainCustDetails\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(8)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bcreateCIS\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(9)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bsearchCIS\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(10)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bpreDEMIValidation\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(11)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bcheckDuplicateApplication\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(12)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\btagForBatchExtract\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(13)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\bgetDEMIDecision\\b.*"))
                .conditionalReplaceValueTransform("MESSAGE",new IntWritable(14)
                        ,new StringRegexColumnCondition("MESSAGE",".*\\bupdateApplicationStatus\\b.*"))
                .conditionalReplaceValueTransform("MESSAGE",new IntWritable(15)
                        ,new StringRegexColumnCondition("MESSAGE",".*\\bupdateStatusNotification\\b.*"))
                     .conditionalReplaceValueTransform("MESSAGE",new IntWritable(16)
                             ,new StringRegexColumnCondition("MESSAGE",".*\\belectronicVerification\\b.*"))


                .stringToTimeTransform("_time","yyyy-MM-dd'T'HH:mm:ss.SSSZ", DateTimeZone.forTimeZone(TimeZone.getTimeZone("AET")))
                 .renameColumn("_time", "DateTime")
                .convertToInteger("PRIORITY")
                .reduce(new Reducer.Builder(ReduceOp.CountUnique)
                        .keyColumns("ARN")
                        .sumColumns("ELAPSEDMS")
                        .maxColumn("declined")
                        .sumColumns("PRIORITY")
                        .maxColumn("DateTime")
                        .minColumns("DateTime")
                        .build())

                .renameColumn("max(DateTime)", "maxTime")
                .renameColumn("min(DateTime)", "minTime")
                .renameColumn("sum(ELAPSEDMS)", "TotalExecutionTime")
                .renameColumn("countunique(MESSAGE)", "noOfStepsExecuted")
                .renameColumn("sum(PRIORITY)", "noOfErrorsOccurred")
                .renameColumn("max(declined)", "notApproved")
                .integerColumnsMathOp("TotalProcessingTime",MathOp.Subtract,"maxTime","minTime")
                .removeColumns("maxTime","minTime")
                .removeColumns("ARN")
                .reorderColumns("noOfStepsExecuted","noOfErrorsOccurred","TotalExecutionTime","TotalProcessingTime","notApproved")
/*
                .reduce(new Reducer.Builder(ReduceOp.CountUnique)
                        .meanColumns("TotalProcessingTime")
                        .build())
*/
            .build();
        Schema outputSchema = tp.getFinalSchema();

        log.info("Schema after transforming data:");
        log.info(outputSchema.toString());

        File inputFile = new ClassPathResource("input/DLASamle.csv").getFile();
        File outputFile = new File("output.csv");
        if(outputFile.exists()){
            outputFile.delete();
        }
        outputFile.createNewFile();

        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(inputFile));

        RecordWriter rw = new CSVRecordWriter();
        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);

        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
        rw.writeBatch(processedData);
        rw.close();


        log.info("\n\n---- Processed Data File ----");
        String fileContents = FileUtils.readFileToString(outputFile);
        log.info("\n\n"+fileContents+"\n\n");

        log.info("\n\nDONE");


    }

}
