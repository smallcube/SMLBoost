package bigcircle;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

//单种分类器抗噪能力的测试

public class noisyTest4{
	
	public static void main(String[] args) throws Exception {
		int m_NumberIterations = 50;
		int classifierType = 0;         //0:j48,  1:ANN,  2:SMO with RBFKernel,  3:DecisionStump
		int repeatedTime = 50;
		double noisyPor = 0.05;
		
		File root = new File("./dataset");
		File[] files = root.listFiles();
		
		String path = "./Noisy/ArcGvUnsupervisedMarginVersion10";
	    File dir = new File(path);
	    if (dir.exists()==false || dir.isDirectory()==false) {
			dir.mkdirs();
		}
		
	    double[] rMyAcc = new double[files.length];
	    double[] rMyMinMargin = new double[files.length];
	    double[] rMyAvgMargin = new double[files.length];
	    double[] rAdaAcc = new double[files.length];
	    double[] rAdaMinMargin = new double[files.length];
	    double[] rAdaAvgMargin = new double[files.length];
	    
	    String[] fileNames = new String[files.length];
	    
		
		for(int f=0;f<files.length;f++){
			int pos = files[f].getName().lastIndexOf(".arff");
		    String fileName = files[f].getName().substring(0, pos);
		    fileNames[f] = fileName;
		    
			ArffLoader arffLoader = new ArffLoader();
			arffLoader.setFile(files[f]);
			Instances dataset = arffLoader.getDataSet();   //读取整个数据集
			dataset.setClassIndex(dataset.numAttributes()-1);
			int nr_fold = 4;
			Random random = new Random(1);
			double avgMyAcc = 0, avgMyMinMargin=0, avgMyAvgMargin=0, avgAdaAcc = 0, avgAdaMinMargin=0, avgAdaAvgMargin=0;
			
			
			double[] foldMyAcc = new double[repeatedTime];
		    double[] foldMyMinMargin = new double[repeatedTime];
		    double[] foldMyAvgMargin = new double[repeatedTime];
		    double[] foldAdaAcc = new double[repeatedTime];
		    double[] foldAdaMinMargin = new double[repeatedTime];
		    double[] foldAdaAvgMargin = new double[repeatedTime];
		    
		    
			for(int i=0;i<repeatedTime;i++){
				Instances randData = new Instances(dataset, 0, dataset.numInstances());
				randData.randomize(random);
				randData.stratify(nr_fold);
		        Instances train1 = randData.trainCV(nr_fold, 0);
		    	Instances test = randData.testCV(nr_fold, 0);
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
		    	Instances train = ig.LabelNoisyGenerate(train1, noisyPor, random);
		    	
		    	ArcGvUnsupervisedMarginVersion10 s = new ArcGvUnsupervisedMarginVersion10(classifier);
			    //AdaBoostM1 s = new AdaBoostM1();
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
					if(instanceJ.classValue() == r){
						thisAcc+=1;
					}
		    	}
			    	
		    	thisAcc /= test.numInstances();
		    	
		    	avgMyAcc += thisAcc;
		    	System.out.println("第"+(i+1) + "重acc=" + thisAcc);
		    	s.getMargin(train);
		    	foldMyAcc[i] = thisAcc;
		    	foldMyMinMargin[i] = s.miniTrueMargin;
		    	foldMyAvgMargin[i] = s.averageTrueMargin;
		    	avgMyMinMargin += s.miniTrueMargin;
		    	avgMyAvgMargin += s.averageTrueMargin;
		    	
		    	//计算AdaBoost相关
		    	
		    	/*
		    	MyAdaBoostM1 s1 = new MyAdaBoostM1();
			    //AdaBoostM1 s = new AdaBoostM1();
			    s1.setClassifier(classifier);
			    s1.setNumIterations(m_NumberIterations);
			    s1.buildClassifier(train);
			    	
		    	thisAcc = 0;
		    	for(int j=0;j<test.numInstances();j++){
		    		Instance instanceJ = test.instance(j);
					double r = s1.classifyInstance(instanceJ);
					if(instanceJ.classValue() == r){
						thisAcc+=1;
					}
		    	}
			    	
		    	thisAcc /= test.numInstances();
		    	avgAdaAcc += thisAcc;
		    	System.out.println("第"+(i+1) + "重AdaBoosting  acc=" + thisAcc);
		    	
		    	s1.getMargin(train);
		    	foldAdaAcc[i] = thisAcc;
		    	foldAdaMinMargin[i] = s1.miniTrueMargin;
		    	foldAdaAvgMargin[i] = s1.averageTrueMargin;
		    	avgAdaMinMargin += s1.miniTrueMargin;
		    	avgAdaAvgMargin += s1.averageTrueMargin;
		    	*/
			}
			/*
			avgAdaAcc /= repeatedTime;
			avgAdaAvgMargin /= repeatedTime;
			avgAdaMinMargin /= repeatedTime;
			*/
			avgMyAcc /= repeatedTime;
			avgMyMinMargin /= repeatedTime;
			avgMyAvgMargin /= repeatedTime;
			
			rMyAcc[f] = avgMyAcc;
		    rMyMinMargin[f] = avgMyMinMargin;
		    rMyAvgMargin[f] = avgMyAvgMargin;
		    rAdaAcc[f] = avgAdaAcc;
		    rAdaMinMargin[f] = avgAdaMinMargin;
		    rAdaAvgMargin[f] = avgAdaAvgMargin;
		    
			
			System.out.println(fileName+"  Myacc="+avgMyAcc+"   AdaAcc="+avgAdaAcc + "   myMinMargin="+avgMyMinMargin + "  AdaMinMargin="+avgAdaMinMargin);
			  
		    FileWriter writer1 = new FileWriter(path+"/Z"+fileName+".txt");
		    writer1.write(fileName+"\n");
		    writer1.write("avgAcc=" + avgMyAcc + "\n");
			for(int i=0;i<repeatedTime;i++){
			    writer1.write(""+foldMyAcc[i]+"\n");
			}
			
			writer1.write("\n\n\nMinMargin=" + avgMyMinMargin + "\n");
			for(int i=0;i<repeatedTime;i++){
			    writer1.write(""+foldMyMinMargin[i]+"\n");
			}
			
			writer1.write("\n\n\nAvgMargin=" + avgMyAvgMargin + "\n");
			for(int i=0;i<repeatedTime;i++){
			    writer1.write(""+foldMyAvgMargin[i]+"\n");
			}
			writer1.flush();
			writer1.close();
			 
			/*
			FileWriter writer3 = new FileWriter(path+"/Ada"+fileName+".txt");
		    writer3.write(fileName+"\n");
		    writer3.write("avgAcc=" + avgAdaAcc + "\n");
			for(int i=0;i<repeatedTime;i++){
			    writer3.write(""+foldAdaAcc[i]+"\n");
			}
			
			writer3.write("\n\n\nMinMargin=" + avgAdaMinMargin + "\n");
			for(int i=0;i<repeatedTime;i++){
			    writer3.write(""+foldAdaMinMargin[i]+"\n");
			}
			
			writer3.write("\n\n\nAvgMargin=" + avgAdaAvgMargin + "\n");
			for(int i=0;i<repeatedTime;i++){
			    writer3.write(""+foldAdaAvgMargin[i]+"\n");
			}
			writer3.flush();
			writer3.close();
			*/
			//写入最后的结果
			
			FileWriter writer2 = new FileWriter(path+"/"+"ZZZAccuracy.txt");
			writer2.write("fileName\n");
			for(int i=0;i<=f;i++){
				writer2.write(fileNames[i]+"\n");
			}
			
			writer2.write("\n\nMyAcc\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rMyAcc[i]+"\n");
			}
			writer2.write("\n\nMyMinMargin\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rMyMinMargin[i]+"\n");
			}
			writer2.write("\n\nMyAvgMargin\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rMyAvgMargin[i]+"\n");
			}
			
			/*
			writer2.write("\n\nAdaAcc\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rAdaAcc[i]+"\n");
			}
			writer2.write("\n\nAdaMinMargin\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rAdaMinMargin[i]+"\n");
			}
			writer2.write("\n\nAdaAvgMargin\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rAdaAvgMargin[i]+"\n");
			}	
			*/
			writer2.flush();
			writer2.close();
		}
	}
		
}