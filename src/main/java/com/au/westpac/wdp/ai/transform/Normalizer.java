package com.au.westpac.wdp.ai.transform;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Normalizer {

    private static Logger log = LoggerFactory.getLogger(Normalizer.class);

    public static void main(String[] args) throws  Exception {


        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("/output.csv").getFile()));
        int labelIndex = 4;
        int numClasses = 2;
        DataSetIterator fulliterator = new RecordReaderDataSetIterator(recordReader,150,labelIndex,numClasses);
        DataSet datasetX = fulliterator.next();


        //log.info("\n{}\n",datasetX);

        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler(0,1);
        preProcessor.fit(datasetX);
        log.info("Transforming a dataset..");
        preProcessor.transform(datasetX);
        log.info("\n{}\n",datasetX.getFeatureMatrix());
        log.info("\n{}\n",datasetX.getLabels());
    }
}
