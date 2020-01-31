package bigcircle;

import java.util.Enumeration;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

//final final final version
public class ArcGvUnsupervisedMarginVersion24{
	protected Classifier m_Classifier = new ZeroR();
	protected Classifier[] m_Classifiers;
	protected int m_NumIterations = 50;
	protected int m_Seed = 1;
	private double minimumMargin = 0;
	protected int mKNN = 5;
	
	private static int MAX_NUM_RESAMPLING_ITERATIONS = 5;
	 
	private double[] m_Betas;
	private Random random = null;
	private double[][] classificationResult = null;
	private double[] sampleWeights = null;
	private double[] noisyValues;
	private double avgNoisyValue;
	
	
	public double miniTrueMargin = 0;
	public double averageTrueMargin = 0;
	public double miniUnsupervisedMargin = 0;
	public double averageTrueMarginVariance = 0;
	public double averageUnsupervisedMargin = 0;
	public double[] margins = null;
	public double[] UnsupervisedMargins = null;
	    
	
	private int[] previousMarginIndex;
	
	private double marginPosThreshold = 0.1;
	
	public ArcGvUnsupervisedMarginVersion24(Classifier classifier){
		m_Classifier = classifier;
	}
	
	public void setNumIterations(int iteration){
		m_NumIterations = iteration;
	}
	
	public void setClassifier(Classifier classifier){
		m_Classifier = classifier;
	}
	
	public void buildClassifier(Instances data) throws Exception{
		
		m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);
		
		init(data);
		
		Instances trainData, sample, training;
	    double epsilon, reweight;
	    int numInstances = data.numInstances();
	    int resamplingIterations = 0;

	    // Initialize data
	    m_Betas = new double [m_Classifiers.length];
	    int m_NumIterationsPerformed = 0;
	    training = new Instances(data, 0, numInstances);
	    for (int i = 0; i < training.numInstances(); i++) {
	    	training.instance(i).setWeight(1.0/data.numInstances());
	    }
	    
	    // Do boostrap iterations
	    for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length; m_NumIterationsPerformed++) {
	    	trainData = new Instances(training);
	         // Resample
	        resamplingIterations = 0;
	        double[] weights = new double[trainData.numInstances()];
	        for (int i = 0; i < trainData.numInstances(); i++) {
	        	weights[i] = trainData.instance(i).weight();
	        	//System.out.println("i=" + i + "   weight="+weights[i]);
	        }
	        do {
	        	sample = GenerateInstances(trainData, weights, random);
	        	m_Classifiers[m_NumIterationsPerformed].buildClassifier(sample);
	        	resamplingIterations++;
	        	
	        	epsilon = 0;
	        	double sumW = 0;
	        	for(int i=0;i<training.numInstances();i++){
	        		sumW += weights[i];
	        		Instance instanceI = training.instance(i);
	        		double tempR = m_Classifiers[m_NumIterationsPerformed].classifyInstance(instanceI);
	        		if(tempR != instanceI.classValue()){
	        			epsilon+= weights[i];
	        		}
	        	}
	        	epsilon /= sumW;
	        } while (Utils.eq(epsilon, 0) && (resamplingIterations < MAX_NUM_RESAMPLING_ITERATIONS));
	      	
	        
	        // Stop if error too big or 0
	        if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
	        	if (m_NumIterationsPerformed == 0) {
	        		m_NumIterationsPerformed = 1; // If we're the first we have to to use it
	        		m_NumIterations = 1;
	        		m_Betas[0] = 1;
	        	}
	        	break;
	        }
	      
	        // Determine the weight to assign to this model
	        //epsilon = 1.0 - 2*epsilon;
	        //m_Betas[m_NumIterationsPerformed] = ( Math.log((1 + epsilon) / (1- epsilon)) + Math.log((1+minimumMargin)/(1-minimumMargin)));
	        m_Betas[m_NumIterationsPerformed] = Math.log((1 - epsilon) / epsilon);
	        reweight = (1 - epsilon) / epsilon;
	        //reweight = m_Betas[m_NumIterationsPerformed];
	        //System.out.println("i=" + m_NumIterationsPerformed+ "   reweight="+reweight + "   epsilon="+epsilon);
	      
	        // Update instance weights
	        setWeights(training, reweight, m_NumIterationsPerformed);
	    }
	    
	    m_NumIterations = m_NumIterationsPerformed;
	}
	
	
	private void setWeights(Instances training, double reweight, int pos) throws Exception{
		double tempMinimumMargin = Double.MAX_VALUE;
		
		double[] currentMargin = new double[training.numInstances()];
		int[] currentMarginIndex = new int[training.numInstances()];
		
		
		for(int i=0;i<training.numInstances();i++){
	    	Instance instance = training.instance(i);
	    	double maxValue=Double.MIN_VALUE, maxValue2=Double.MIN_VALUE,sums=0;
			int maxIndex = 0, maxIndex2 = 0;
			
			double r = m_Classifiers[pos].classifyInstance(instance);
	    	classificationResult[i][(int)r] += m_Betas[pos];
	    	
	    	maxValue=Double.MIN_VALUE;
	    	maxValue2=Double.MIN_VALUE;
	    	sums=0;
			maxIndex = 0;
			maxIndex2 = 0;
			
			for(int j=0;j<training.numClasses();j++){
				sums+=classificationResult[i][j];
				if(classificationResult[i][j] > maxValue){
					maxValue = classificationResult[i][j];
					maxIndex = j;
				}
			}
			
			for(int j=0;j<training.numClasses();j++){
				if((j!=maxIndex) && (classificationResult[i][j] > maxValue2)){
					maxValue2 = classificationResult[i][j];
					maxIndex2 = j;
				}
			}
			currentMargin[i] = (maxValue-maxValue2)/sums;
			
			if (tempMinimumMargin > currentMargin[i]) {
				tempMinimumMargin = currentMargin[i];
			}
	    }
		
		boolean[] isUsed = new boolean[training.numInstances()];
		for(int i=0;i<isUsed.length;i++){
			isUsed[i] = false;
		}
		for(int i=0;i<training.numInstances();i++){
			int count = 1;
			for(int j=0;j<training.numInstances();j++){
				if (j==i) {
					continue;
				}
				if((currentMargin[j]<currentMargin[i]) || (currentMargin[j] == currentMargin[i] && isUsed[j])){
					count++;
				}
			}
			isUsed[i] = true;
			currentMarginIndex[i] = count;
			
			//reweight: the i-th instance has been pushed to the decision boundary
			double threhold = 1.0*currentMarginIndex[i] / training.numInstances();
			if(currentMarginIndex[i] < previousMarginIndex[i] && threhold < this.marginPosThreshold && noisyValues[i]<=avgNoisyValue){
				double tempW = training.instance(i).weight();
				//training.instance(i).setWeight(tempW*Math.exp(reweight));
				training.instance(i).setWeight(tempW*reweight);
			}
		}
		
		minimumMargin = tempMinimumMargin;
		if (minimumMargin == 1) {
			minimumMargin = 0;
		}
	    // Renormalize weights
	    double newSumOfWeights = training.sumOfWeights();
	    Enumeration enu = training.enumerateInstances();
	    while (enu.hasMoreElements()) {
	       Instance instance = (Instance) enu.nextElement();
	       instance.setWeight(instance.weight() / newSumOfWeights);
	    }
	    
	    for(int i=0;i<previousMarginIndex.length;i++){
	    	previousMarginIndex[i] = currentMarginIndex[i];
	    }
	}
	
	private Instances GenerateInstances(Instances data, double[] weights, Random random){
		int[] sampled = new int[data.numInstances()];
		if (weights.length != data.numInstances()) {
			throw new IllegalArgumentException("weights.length != numInstances.");
		}

		Instances newData = new Instances(data, data.numInstances());
		if (data.numInstances() == 0) {
		     return newData;
		}

	    // Walker's method, see pp. 232 of "Stochastic Simulation" by B.D. Ripley
	    double[] P = new double[weights.length];
	    System.arraycopy(weights, 0, P, 0, weights.length);
	    Utils.normalize(P);
	    double[] Q = new double[weights.length];
	    int[] A = new int[weights.length];
	    int[] W = new int[weights.length];
	    int M = weights.length;
	    int NN = -1;
	    int NP = M;
	    for (int I = 0; I < M; I++) {
           if (P[I] < 0) {
	         throw new IllegalArgumentException("Weights have to be positive.");
	       }
	       Q[I] = M * P[I];
	       if (Q[I] < 1.0) {
	         W[++NN] = I;
	       }
	       else {
	         W[--NP] = I;
	       }
	    }
	    if (NN > -1 && NP < M) {
	       for (int S = 0; S < M - 1; S++) {
	          int I = W[S];
	          int J = W[NP];
	          A[I] = J;
	          Q[J] += Q[I] - 1.0;
	          if (Q[J] < 1.0) {
	            NP++;
	          }
	          if (NP >= M) {
	            break;
	          }
	       }
	      // A[W[M]] = W[M];
	    }

	    for (int I = 0; I < M; I++) {
	      Q[I] += I;
	    }
	    
	    for (int i = 0; i < data.numInstances(); i++) {
			sampled[i] = 0;
	    }
		for (int i = 0; i < data.numInstances(); i++) {
			int ALRV;
		    double U = M * random.nextDouble();
		    int I = (int) U;
		    if (U < Q[I]) {
		        ALRV = I;
		    }
		    else {
		        ALRV = A[I];
		    }
		    newData.add(data.instance(ALRV));
		    if (sampled != null) {
		        sampled[ALRV]++;
		    }
		    newData.instance(newData.numInstances() - 1).setWeight(1);
		}
		
		return newData;
	}
	
	protected int selectIndexProbabilistically(double []cdf){
		double rnd = random.nextDouble();
		int index = 0;
		while(index < cdf.length && rnd > cdf[index]){
		    index++;
		}
		return index;
	} 
	
	public double classifyInstance(Instance instance) throws Exception {
	    double[] dist = distributionForInstance(instance);
	    if (dist == null) {
	        throw new Exception("Null distribution predicted");
	    }
	    switch (instance.classAttribute().type()) {
	    case Attribute.NOMINAL:
	       double max = 0;
	       int maxIndex = 0;

	       for (int i = 0; i < dist.length; i++) {
	          if (dist[i] > max) {
	             maxIndex = i;
	             max = dist[i];
	          }
	       }
	       if (max > 0) {
	           return maxIndex;
	       }
	       else {
	    	   return Instance.missingValue();
	       }
	    case Attribute.NUMERIC:
	    case Attribute.DATE:
	    	return dist[0];
	    default:
	    	return Instance.missingValue();
	    }
	}
	
	/**
	   * calculates the class membership probabilities for the given test instance
	   *
	   * @param instance: the instance needs to be classified
	   * @return predicted class probability distribution
	   * @throws Exception if something goes wrong
	   */
	public double[] distributionForInstance(Instance instance) throws Exception{
		double[] sums = new double[instance.numClasses()];
		double[] newProbs;
		for (int i = 0; i < m_NumIterations; i++) {
			if (instance.classAttribute().isNumeric() == true) {
				sums[0] += m_Classifiers[i].classifyInstance(instance);
			}
			else{
				//System.out.println("i="+i);
				newProbs = m_Classifiers[i].distributionForInstance(instance);
				/*
				for(int j=0;j<newProbs.length;j++){
					//sums[j] += newProbs[j];
					sums[j] += newProbs[j]*m_Betas[i];
				}
				*/
				sums[(int)m_Classifiers[i].classifyInstance(instance)] += m_Betas[i];
			}
		}
		//return Utils.logs2probs(sums);
		//System.out.println("sumLen="+sums.length);
		return sums;
		/*
		if (instance.classAttribute().isNumeric() == true) {
			sums[0] /= (double)m_NumIterations;
			return sums;
		}
		else if (Utils.eq(Utils.sum(sums), 0)) {
			return sums;
		}
		else{
			Utils.normalize(sums);
			return sums;
		}
		*/
	}
	
	public void getMargin(Instances data) throws Exception{
		this.margins = new double[data.numInstances()];
	    this.UnsupervisedMargins = new double[data.numInstances()];
	    
		double minValue = Double.MAX_VALUE, minUnsupervisedValue=Double.MAX_VALUE;
		double avgMargin = 0, avgUnsupervisedMargin = 0;;
		for(int i=0;i<data.numInstances();i++){
			double sum = 0;
			double[] sums = new double[data.numClasses()];
			Instance instanceI = data.instance(i);
			for(int j=0;j<m_NumIterations;j++){
				sums[(int)m_Classifiers[j].classifyInstance(instanceI)] += m_Betas[j];
			}
			
			double maxValue=Double.MIN_VALUE;
	    	double maxValue2=Double.MIN_VALUE;
	    	
			int maxIndex = 0;
			int maxIndex2 = 0;
			
			for(int j=0;j<sums.length;j++){
				sum += sums[j];
				if (sums[j] > maxValue) {
					maxValue = sums[j];
					maxIndex = j;
				}
			}
			
			for(int j=0;j<sums.length;j++){
				if (j!=maxIndex && sums[j] > maxValue2) {
					maxValue2 = sums[j];
					maxIndex2 = j;
				}
			}
			
			double unsuperVisedMargin = (maxValue-maxValue2)/sum;
			this.UnsupervisedMargins[i] = unsuperVisedMargin;
			avgUnsupervisedMargin += unsuperVisedMargin;
			if (unsuperVisedMargin < minUnsupervisedValue) {
				minUnsupervisedValue = unsuperVisedMargin;
			}
			
			double margin = 2*sums[(int)instanceI.classValue()] / sum - 1;
			this.margins[i] = margin;
			avgMargin += margin;
			if(margin < minValue){
				minValue = margin;
			}
		}
		
		avgMargin /= data.numInstances();
		avgUnsupervisedMargin /= data.numInstances();
		//System.out.println("Classifier Number=" + m_NumIterationsPerformed + "  avgMargin="+avgMargin + "  m_Betas[0]=" + m_Betas[0]);
		
		this.miniTrueMargin = minValue;
		this.averageTrueMargin = avgMargin;
		this.miniUnsupervisedMargin = miniUnsupervisedMargin;
		this.averageUnsupervisedMargin = avgUnsupervisedMargin;
		this.averageTrueMarginVariance = 0;
		
		for(int i=0;i<this.margins.length;i++){
			this.averageTrueMarginVariance += Math.pow(this.averageTrueMargin-this.margins[i], 2);
		}
		
		this.averageTrueMarginVariance /= this.margins.length;
		
	}
	
	
	public double getTestAcc(Instances data, int num) throws Exception{
		double thisAcc = 0;
		for(int i=0;i<data.numInstances();i++){
			Instance instanceI = data.instance(i);
			double sum = 0;
			double[] sums = new double[data.numClasses()];
			
			for(int j=0;j<num&&j<m_NumIterations;j++){
				//System.out.println("number="+num);
				sums[(int)m_Classifiers[j].classifyInstance(instanceI)] += m_Betas[j];
			}
			
			double maxValue = Double.MIN_VALUE;
			int maxIndex = -1;
			for(int j=0;j<sums.length;j++){
				if(sums[j] > maxValue){
					maxValue = sums[j];
					maxIndex = j;
				}
			}
			
			if(maxIndex == instanceI.classValue()){
				thisAcc += 1;
			}
		}
		thisAcc /= data.numInstances();
		return thisAcc;
	}
	
	//used to initnize the parameters
	private void init(Instances data) throws Exception{
		random = new Random(m_Seed);
		minimumMargin = 0;
		
		classificationResult = new double[data.numInstances()][data.numClasses()];
		for(int i=0;i<data.numInstances();i++){
			for(int j=0;j<data.numClasses();j++){
				classificationResult[i][j] = 0;
			}
		}
		
		previousMarginIndex = new int[data.numInstances()];
		for(int i=0;i<previousMarginIndex.length;i++){
			previousMarginIndex[i] = i+1;
		}
		
		noisyValues = new double[data.numInstances()];
		avgNoisyValue = 0;
		
		MyIBk myIBk = new MyIBk();
		myIBk.setKNN(mKNN);
		myIBk.buildClassifier(data);
		this.marginPosThreshold = 0;
		for(int i=0;i<data.numInstances();i++){
			Instance instanceI = data.instance(i);
			Instances neighbors = myIBk.getNearestNeighbors(instanceI);
			double count = 0;
			for(int j=0;j<neighbors.numInstances();j++){
				Instance instanceJ = neighbors.instance(j);
				if(instanceI.classValue() != instanceJ.classValue()){
					count += 1;
				}
			}
			noisyValues[i] = count/neighbors.numInstances();
			avgNoisyValue += noisyValues[i];
			
			if (count > mKNN/2.0 && count <mKNN) {
				this.marginPosThreshold += 1.0;
			}
		}
		avgNoisyValue /= data.numInstances();
		this.marginPosThreshold /= data.numInstances();
		
		//System.out.println("MarginThreshold="+marginPosThreshold);
	}
}