package bigcircle;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.SimpleCart;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class mainTest{
	
	public static void main(String[] args) throws Exception {
		int m_NumberIterations = 50;
		int classifierType = 0;         //0:j48,  1:ANN,  2:SMO with RBFKernel,  3:DecisionStump
		
		File root = new File("./dataset");
			File[] files = root.listFiles();
			
			String path = "./FinalResult/ArcGvUnsupervisedMarginVersion4";
		    File dir = new File(path);
		    if (dir.exists()==false || dir.isDirectory()==false) {
				dir.mkdirs();
			}
			
		    double[] rFmeasure = new double[files.length];
		    double[] rAuc = new double[files.length];
		    double[] rAcc = new double[files.length];
		    double[] rGmean = new double[files.length];
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
			    
			    double[] fold_acc = new double[nr_fold];
			    double[] fold_Auc = new double[nr_fold];
			    double[] fold_Fmeasure = new double[nr_fold];
			    double[] fold_Gmean = new double[nr_fold];
			    double avgAcc = 0, avgAuc=0, avgFmeasure=0, avgGmean=0;
			    
			    
			    double[] predictResult = new double[randData.numInstances()];
				FastVector mPredictions = new FastVector();
				double TP = 0;         //True positive
			    double FP = 0;         //False positive
			    double TN = 0;         //True necgtive
			    double FN = 0;         //False necgtive
			    double pNumber = 0;    //number of positive samples
			    double nNumber = 0;    //number of necgtive samples
			    double precision = 0;  //measure of precision
			    double recall = 0;     //measure of recall
			    double Gmean = 0;
			    ThresholdCurve tc = new ThresholdCurve();
			    
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
			    	else if (classifierType==4){
			    		classifier = new SimpleCart();
			    	}
			    	
			    	//BaggingWithTrueMargin s = new BaggingWithTrueMargin(classifier);
			    	//wBaggingWithPruning s = new wBaggingWithPruning(classifier);
			    	//wBaggingWithPruning s = new wBaggingWithPruning(classifier);
			    	ArcGvUnsupervisedMarginVersion4 s = new ArcGvUnsupervisedMarginVersion4(classifier);
			    	//AdaBoostM1 s = new AdaBoostM1();
			    	//s.setUseResampling(true);
			    	s.setClassifier(classifier);
			    	s.setNumIterations(m_NumberIterations);
			    	s.buildClassifier(train);
			    	//s.pruningWithValidation2(s.test);
			    	//s.BackPruning(s.test);
			    	//s.pruning();
			    	//minorityLabel = s.minorityLabel;
			    	
			    	double thisAcc = 0;
			    	for(int j=0;j<test.numInstances();j++){
			    		Instance instanceJ = test.instance(j);
						double r = s.classifyInstance(instanceJ);
						predictResult[predictPos] = r;
						predictPos++;
						double[] dist = s.distributionForInstance(instanceJ);
						mPredictions.addElement(new NominalPrediction(instanceJ.classValue(), dist, instanceJ.weight()));
						
						if(instanceJ.classValue() == minorityLabel){
							pNumber+=1;
							if (instanceJ.classValue() == r){
				    			TP += 1;
				    		}
				    		else{
				    			FN += 1;
				    		}
						}
						else{
							nNumber+=1;
							if (instanceJ.classValue() == r){
				    			TN += 1;
				    		}
				    		else{
				    			FP += 1;
				    		}
						}
						if(instanceJ.classValue() == r){
							thisAcc+=1;
						}
			    	}
			    	
			    	thisAcc /= test.numInstances();
			    	fold_acc[i] = thisAcc;
			    	avgAcc += thisAcc;
			    	System.out.println("第"+(i+1) + "重acc=" + fold_acc[i]);
			    	
			    }
			    
			  //计算这一折的AUC
		    	Instances result = tc.getCurve(mPredictions, (int)minorityLabel);
			    double auc = ThresholdCurve.getROCArea(result);
			    rAuc[f] = auc;
			    //计算Fmeasure
			    precision = TP/(TP+FP);
			    recall = TP/(TP+FN);
			    rFmeasure[f] = 2*precision*recall/(precision+recall);
			    //计算Gmean
			    double TPrate = TP / (TP + FN);
			    double TNrate = TN / (TN + FP);
			    rGmean[f] = Math.sqrt(TPrate * TNrate);
			    
			    avgAcc /= nr_fold;
			    rAcc[f] = avgAcc;  
			    fileNames[f] = fileName;
			    
			    System.out.println(fileName+"  acc="+avgAcc+"   auc="+auc);
			    
			    FileWriter writer1 = new FileWriter(path+"/"+fileName+".txt");
			    writer1.write(fileName+"\n");
			    writer1.write("avgAcc=" + avgAcc + "\n");
			    for(int i=0;i<nr_fold;i++){
			    	writer1.write(""+fold_acc[i]+"\n");
			    }
			    
			    writer1.flush();
			    writer1.close();
			    
			    
			    FileWriter writer2 = new FileWriter(path+"/"+"ZZZAccuracy.txt");
				writer2.write("fileName\n");
				for(int i=0;i<=f;i++){
					writer2.write(fileNames[i]+"\n");
				}
				writer2.write("\n\nAcc\n");
				for(int i=0;i<=f;i++){
					writer2.write(""+rAcc[i]+"\n");
				}
				writer2.write("\n\nAUC\n");
				for(int i=0;i<=f;i++){
					writer2.write(""+rAuc[i]+"\n");
				}
				writer2.write("\n\nFmeasure\n");
				for(int i=0;i<=f;i++){
					writer2.write(""+rFmeasure[i]+"\n");
				}
				writer2.write("\n\nGmean\n");
				for(int i=0;i<=f;i++){
					writer2.write(""+rGmean[i]+"\n");
				}
				
				writer2.flush();
				writer2.close();
			}
		}
		
}