package bigcircle;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public class InstanceGenerate{
	public int nearestNeighbors = 5;
	public int k2 = 5;
	public double[] artWeight;
	public double lamda = 0.8;
	public double minorityLabel = 0;
	
	
	public InstanceGenerate(){
		
	}
	
	public Instances SMOTE(Instances data){
		
		
		return data;
	}
	
	
	protected Integer[] randomReplace(Integer[] indices, int replaceSize, Random random) {
		Integer[] result = new Integer[replaceSize];
		Collections.shuffle(Arrays.asList(indices), random);
		for (int i=0;i<replaceSize;i++){
			result[i] = indices[i];
		}
		Collections.sort(Arrays.asList(result));
		return result;
	}
	
	public Instances LabelNoisyGenerate(Instances data, double ratio, Random random){
		Integer[] index = new Integer[data.numInstances()];
		for(int i=0;i<index.length;i++){
			index[i] = i;
		}
		Collections.shuffle(Arrays.asList(index), random);
		if (ratio >1.0) {
			ratio = 1.0;
		}
		if(ratio<0){
			ratio = 0;
		}
		int len = (int)Math.round(data.numInstances()*ratio);
		
		Instances result = new Instances(data, 0, 0);
		for(int i=0;i<data.numInstances();i++){
			Instance instanceI = data.instance(i);
			
			if(i<len){
				int r = random.nextInt(data.numClasses());
				if(r==instanceI.classValue()){
					r = (r+1)%data.numClasses();
				}
				instanceI.setClassValue(r);
			}
			result.add(instanceI);
		}
		return result;
	}
}