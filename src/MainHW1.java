import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Instances generate1DLinData(int numInstances) {
        Attribute Attribute1 = new Attribute("x");
        Attribute ClassAttribute = new Attribute("y");
        FastVector fvWekaAttributes = new FastVector(2);
        fvWekaAttributes.addElement(Attribute1);
        fvWekaAttributes.addElement(ClassAttribute);
        Instances data = new Instances("Rel", fvWekaAttributes, numInstances);
        data.setClassIndex(1);
        // y = 3x
        for (int i = 0; i < numInstances; i++) {
            Instance iExample = new Instance(2);
            int x = ThreadLocalRandom.current().nextInt(0, 100);
            int y = x * 3;
            iExample.setValue((Attribute) fvWekaAttributes.elementAt(0), x);
            iExample.setValue((Attribute) fvWekaAttributes.elementAt(1), y);
            data.add(iExample);
        }

        return data;

    }

    /**
     * Sets the class index as the last attribute.
     *
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {
        //todo load data
        String filename = "housing_training.txt";

        BufferedReader inputReader = null;

//        read data file
        inputReader = readDataFile(filename);

        Instances trainingData = new Instances(inputReader);

        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        LinearRegression predictor = new LinearRegression();

        //train the classifier using the algorithm described in the TXT file
        predictor.buildClassifier(trainingData);

        //calculate the SE of the classifier
        predictor.calculateSE(trainingData);


        //test on housing testing
//        String testing = "housing_testing.txt";

//        inputReader = null;

//        //read data file
//        inputReader = readDataFile(testing);
//
//        Instances testingData = new Instances(inputReader);
//
//        testingData.setClassIndex(testingData.numAttributes() - 1);

        //predictor = new LinearRegression();
//
//        for(int q = 0 ; q < testingData.numInstances() ; q++){
//            System.out.println(predictor.regressionPrediction(testingData.instance(q)) - testingData.instance(q).classValue());
//        }
//

        // training on linear data function

//        Instances benchmark = generate1DLinData(2000);
//
//        benchmark.setClassIndex(1);
//        System.out.println(benchmark.toSummaryString());
//        System.out.println(benchmark.toString());
//
//        LinearRegression predictor = new LinearRegression();
//
//        predictor.buildClassifier(benchmark);
//        predictor.calculateSE(benchmark);

        predictor.writeToTXT();
    }


}
