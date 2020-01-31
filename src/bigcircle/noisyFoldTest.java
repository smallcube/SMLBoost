package bigcircle;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class noisyFoldTest{
	
	public static void main(String[] args) throws Exception {
		int m_NumberIterations = 50;
		int classifierType = 0;         //0:j48,  1:ANN,  2:SMO with RBFKernel,  3:DecisionStump
		double noisyPor = 0.00;
		
		File root = new File("./dataset1");
		File[] files = root.listFiles();
		
		String path = "./FinalResult/ArcGvUnsupervisedMarginVersion25+"+noisyPor;
	    File dir = new File(path);
	    if (dir.exists()==false || dir.isDirectory()==false) {
			dir.mkdirs();
		}
		
	    double[] rAccMy = new double[files.length];
	    double[] rAccBoosting = new double[files.length];
	    String[] fileNames = new String[files.length];
		
		for(int f=0;f<files.length;f++){
			int pos = files[f].getName().lastIndexOf(".arff");
		    String fileName = files[f].getName().substring(0, pos);
		    
			ArffLoader arffLoader = new ArffLoader();
			arffLoader.setFile(files[f]);
			Instances dataset = arffLoader.getDataSet();   //读取整个数据集
			dataset.setClassIndex(dataset.numAttributes()-1);
			int nr_fold = 10;
			Random random = new Random(1);
		    Instances randData = new Instances(dataset, 0, dataset.numInstances());
		    randData.randomize(random);
		    randData.stratify(nr_fold);
		    
		    //file 1: experimental results of my own ensemble learning
		   
		    double minorityLabel = 0;
		    
		    int predictPos = 0;
		    
		    double[] fold_acc_my = new double[nr_fold];
		    double[] fold_acc_Boosting = new double[nr_fold];
		    double avgAccMy = 0, avgAccBoosting=0, avgFmeasure=0, avgGmean=0;
		    
		    
		    for (int i=0;i<nr_fold;i++){
		    	Instances train = randData.trainCV(nr_fold, i);
		    	Instances test = randData.testCV(nr_fold, i);
		    	Classifier classifier = null;
		    	if (classifierType==0) {
		    		J48 j48 = new J48();
		    		//j48.setUnpruned(false);
		    		classifier = j48;
				}
		    	else if (classifierType==1) {
		    		classifier = new MultilayerPerceptron();
				}
		    	else if (classifierType==2) {
					SMO smo = new SMO();
					RBFKernel kernel = new RBFKernel();
					smo.setKernel(kernel);
					classifier = smo;
				}
		    	else if (classifierType==3) {
					classifier = new DecisionStump();
				}
		    	
		    	InstanceGenerate ig = new InstanceGenerate();
		    	Instances train1 = ig.LabelNoisyGenerate(train, noisyPor, random);
		    	//NDBoost s = new NDBoost(classifier);
		    	//BaggingWithTrueMargin s = new BaggingWithTrueMargin(classifier);
		    	//wBaggingWithPruning s = new wBaggingWithPruning(classifier);
		    	//wBaggingWithPruning s = new wBaggingWithPruning(classifier);
		    	//ArcGvUnsupervisedMarginVersion11 s = new ArcGvUnsupervisedMarginVersion11(classifier);
		    	//RandomForest s = new RandomForest();
		    	//s.setNumTrees(m_NumberIterations);
		    	//AdaBoostM1 s = new AdaBoostM1();
		    	//ArcGvUnsupervisedMarginVersion25 s = new ArcGvUnsupervisedMarginVersion25(classifier);
		    	//ArcGvUnsupervisedMarginVersion17 s = new ArcGvUnsupervisedMarginVersion17(classifier);
		    	LogitBoost s = new LogitBoost();
		    	s.setClassifier(classifier);
		    	s.setNumIterations(m_NumberIterations);
		    	//J48 s = new J48();
		    	s.buildClassifier(train1);
		    	
		    	double thisAcc = 0;
		    	for(int j=0;j<test.numInstances();j++){
		    		Instance instanceJ = test.instance(j);
					double r = s.classifyInstance(instanceJ);
					
					if(instanceJ.classValue() == r){
						thisAcc+=1;
					}
		    	}
		    	
		    	thisAcc /= test.numInstances();
		    	fold_acc_my[i] = thisAcc;
		    	avgAccMy += thisAcc;
		    	System.out.println("第"+(i+1) + "重acc=" + fold_acc_my[i]);
		    	
		    }
		    
		  //计算这一折的AUC
	    	
		    avgAccMy /= nr_fold;
		    rAccMy[f] = avgAccMy;  
		    fileNames[f] = fileName;
		    
		    System.out.println(fileName+"  acc="+avgAccMy);
		    
		    FileWriter writer1 = new FileWriter(path+"/My"+fileName+".txt");
		    writer1.write(fileName+"\n");
		    writer1.write("avgAcc=" + avgAccMy + "\n");
		    for(int i=0;i<nr_fold;i++){
		    	writer1.write(""+fold_acc_my[i]+"\n");
		    }
		    
		    writer1.flush();
		    writer1.close();
		    
		    
		    FileWriter writer2 = new FileWriter(path+"/"+"ZZZMyAccuracy.txt");
			writer2.write("fileName\n");
			for(int i=0;i<=f;i++){
				writer2.write(fileNames[i]+"\n");
			}
			writer2.write("\n\nAcc\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rAccMy[i]+"\n");
			}
			
			writer2.flush();
			writer2.close();
			
		}
	}
		
}