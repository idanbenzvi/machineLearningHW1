import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SystemInfo;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class LinearRegression extends Classifier{

	private final int MIN_ALPHA_POW = -17;
	private final int MAX_ALPHA_POW = 2;
	private final double MIN_SIG_CHANGE = 0.003;
	private int TRAINING_ITERATIONS = 20000;

	private boolean trainingSwitch;
	private double bestAlpha;
	private double bestSE = Double.MAX_VALUE; //init to very high value that will be changed on the first iteration

    private Instances m_means;

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private int m_numInstances;
	private double[] m_coefficients;

    private double m_alpha;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		//since class attribute is also an attribuite we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;
		m_numInstances = trainingData.numInstances();

        // in order to increase efficiency, we first standardize the data to mean 0 and [-1,1] range
        m_means = standardizeData(trainingData);

		//temp init for m_coefficients
		m_coefficients = new double[m_truNumAttributes+1];

        //train the model and return the best alpha value
		setAlpha(trainingData);

        //perform gradient descent process until convergence (until threshold limit is achieved)
		m_coefficients = gradientDescent(trainingData);

        //for testing only
//        System.out.println(m_alpha + " " + Arrays.toString(m_coefficients));
    }

    /** This function is used to standardize the dataset in order to enable quicker and more efficient calcualations
    * @Instances data - dataset loaded previoysly
    * @throws Exception*/
    public Instances standardizeData(Instances data) throws Exception
    {
        double means[] = new double[m_truNumAttributes];
        double tempSum =0;

        Standardize norm = new Standardize();
        norm.setInputFormat(data);
        Instances processed_training_data = Filter.useFilter(data, norm);

        return processed_training_data;
    }

	private void setAlpha(Instances trainingData) throws Exception {
        trainingSwitch = true;

        for(int i = 0 ; i < m_truNumAttributes+1 ; i++){
            m_coefficients[i] = 1;
        }

        // iterate through a range of possible alpha values and choose the best one generating lowest cost (J) function
        for(int i = MIN_ALPHA_POW ; i < MAX_ALPHA_POW ; i++){
            m_alpha = Math.pow(3,i);
            m_coefficients = gradientDescent(trainingData);

            // for testing only!
//            System.out.println(Arrays.toString(m_coefficients));
        }

        //exit the training mode
        trainingSwitch = false;

        //set best alpha
        m_alpha = bestAlpha;

	}
	
	/**
	 * An implementation of the gradient descent algorithm which should try
	 * to return the weights of a linear regression predictor which minimizes
	 * the average squared error.
	 * @param trainingData
	 * @return
	 * @throws Exception
	 */
	public double[] gradientDescent(Instances trainingData)
			throws Exception {

		double previousSE;
		double curSE;
		double innerProduct;
		double storeSE;
		int iteration;

		double[] regressionWeights = new double[m_truNumAttributes+1]; //+1 for theta0
		double[] tempCalc ;
		double[] tempWeights = new double[m_truNumAttributes+1];
		double delta = MIN_SIG_CHANGE +1; //for the first run


        //init the regression weights
        for(int i = 0 ; i < m_truNumAttributes+1 ; i++){
            regressionWeights[i] = 1;
        }
        regressionWeights[0] = 0;

		previousSE = 0;

		iteration = 0;

		//iterate as long as the gradient changes within given significance boundries
		while(true ) {

            //init temp array
            tempCalc = new double[m_truNumAttributes+1];

			//another iteration to be added
			iteration++;

            //stop condition
            if(trainingSwitch == false)
                if(iteration % 100 == 0 && delta>MIN_SIG_CHANGE)
                return regressionWeights;

                    //go over all attributes of the given dataset
			for (int z = 0; z < m_truNumAttributes+1; z++) {
                //per each instance, calculate the inner product and deduce the yj and multiply by xj (as formulated)
				for (int k = 0; k < m_numInstances; k++) {

                    innerProduct = 0;

					Instance curInstance = trainingData.instance(k);

					//calculate inner product
					for (int i = 1; i <= m_truNumAttributes; i++) {
						if (i != m_ClassIndex+1)
							innerProduct += curInstance.value(i - 1) * regressionWeights[i];
					}
					innerProduct += regressionWeights[0];

					//calculate the gradient for this attribute
					if (z == 0) //for theta zero gradient
						tempCalc[z] += innerProduct - curInstance.classValue();
					else
						tempCalc[z] += (innerProduct - curInstance.classValue()) * curInstance.value(z);
				}
				//calculate the temp weights - store in a temporary array and replace values later.
				tempWeights[z] = regressionWeights[z] - (m_alpha * tempCalc[z] / m_numInstances) ;
			}


            //System.out.println(Arrays.toString(tempWeights));

            //after comletion of run over all weights - update
			regressionWeights = tempWeights;
			//required to evaluate the current SE and predict values during training
			m_coefficients = regressionWeights;

			curSE = calculateSE(trainingData);

			//if we are still training the model - let it run until it reaches the max. allowed iterations
			if(trainingSwitch) {
				delta = MIN_SIG_CHANGE + 1; // by adding +1 we can keep on iterating, through the while loop

				//training only
				if(iteration==TRAINING_ITERATIONS) {

                    // calculate the SE (J function) and evaluate its value - if its smaller than previous values of error,
                    // select the alpha value to be assigned as the best alpha yet.
						if(curSE<bestSE ) {
						bestSE = curSE;
						bestAlpha = m_alpha;

						System.out.println("CURRENT:");
						System.out.println(bestAlpha);
						System.out.println(bestSE);
					}

					return regressionWeights;
				}
			}
			else
            // if not in training mode - simply calculate the current delta between curSE and the one from previous 100 iterations
				delta = Math.abs(curSE - previousSE);

            //make the current iteration's error value the one to compare to within 100 iterations
			if(iteration % 100 == 0)
                previousSE = curSE;


		}
	//return regressionWeights;
	}
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by coefficients on a single instance.
	 * @param instance
	 * @param coefficients
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double prediction = 0.0 ;

		for(int k = 0 ; k < m_truNumAttributes ; k++ )
		{
			if(k!=m_ClassIndex+1)
			prediction = prediction + m_coefficients[k+1] * instance.value(k);
		}

		prediction = prediction + m_coefficients[0];

		return prediction;
	}
	
	/**
	 * Calculates the total squared error over the test data on a linear regression
	 * predictor with weights given by coefficients.
	 * @param testData
	 * @param coefficients
	 * @return
	 * @throws Exception
	 */
	public double calculateSE(Instances testData) throws Exception {
		//iterate over all instances and use their attribute values in order to calculate the J function
		double tempSum = 0;

        //implement the hypthoses  - class value ^ 2 formula to calculate the cost (error) given current thata values
		for(int i = 0 ; i < m_numInstances; i++)
		{
			tempSum += Math.pow((regressionPrediction(testData.instance(i)) - testData.instance(i).classValue()),2);
        }

        double finalSum = tempSum / m_numInstances;

		return finalSum;
	}

    /**
     * Assistive function for homework output
     */
	public void writeToTXT() {
		try{
			String weightsText = "The weights for question 7 are: ";
			String SEText = "The error for question 7 is ";

			File file = new File("hw1.txt");

			// if file doesnt exists, then create it
			if (!file.exists()) {
				file.createNewFile();
			}

			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			bw.write(weightsText + Arrays.toString(m_coefficients));
			bw.newLine();
			bw.write(SEText + m_alpha);
			bw.close();

			//output onto txt file done
		}
		catch(IOException e) {
			e.printStackTrace();

		}
	}


	/**
	 * Finds the closed form solution to linear regression with one variable.
	 * Should return the coefficient that is to be multiplied
	 * by the input attribute.
	 * @param data
	 * @return
	 */
	public double findClosedForm1D(Instances data){
		return 0;
	}

}
