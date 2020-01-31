package bigcircle;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class BorderLine {
	public int nearestNeighbors = 5;
	public int k2 = 5;
	public double[] artWeight;
	public double lamada = 0.5;
	public BorderLine(){
		;
	}
	
	public Instances Sample(Instances prob){
		//使用BorderLine
		//prob:  数据集
		//number:生成的少类样本的数量
		Random rand = new Random(1);
		Instances artInstance = new Instances(prob, 0, 0);
		ArrayList<Instances> inst = new ArrayList<>();
		ArrayList<Instances> borderInst = new ArrayList<>();
		
		for(int i=0;i<prob.numInstances();i++){
			boolean isFind = false;
			for(int j=0;j<inst.size();j++){
				if (inst.get(j).instance(0).classValue() == prob.instance(i).classValue()) {
					inst.get(j).add(prob.instance(i));
					isFind = true;
					break;
				}
			}
			if (!isFind) {
				Instances temp = new Instances(prob, i, 1);
				inst.add(temp);
			}
			
			Instance instanceI = prob.instance(i);
			List distanceToInstance = new LinkedList();
			
			for (int j = 0; j < prob.numInstances(); j++) {
				if (i==j) {
					continue;
				}
				Instance instanceJ = prob.instance(j);
				double distance = 0;
				for(int index=0;index<prob.numAttributes()-1;index++){
					double jVal = instanceJ.value(index);
		            double iVal = instanceI.value(index);
		            distance += Math.pow(jVal-iVal, 2);
				}
				distance = Math.pow(distance, 0.5);
				distanceToInstance.add(new Object[] { distance, instanceJ });
			}
			
			//sort the instances according their distance to i-th sample
			Collections.sort(distanceToInstance, new Comparator() {
		        public int compare(Object o1, Object o2) {
		          double distance1 = (Double) ((Object[]) o1)[0];
		          double distance2 = (Double) ((Object[]) o2)[0];
		          return Double.compare(distance1, distance2);
		        }
		      });
			
			//get the k nearest samples of i-th sample
			double thisSafeLevel = 0;
			for(int j=0;j<nearestNeighbors;j++){
				Instance temp = (Instance)(((Object[])distanceToInstance.get(j))[1]);
				double thisDistance = (Double)(((Object[])distanceToInstance.get(j))[0]);
				
		    	if (temp.classValue()!=instanceI.classValue()) {
		    		thisSafeLevel+=1;
				}
			}
			
			//borderline sample
			if (thisSafeLevel >= nearestNeighbors/2 && thisSafeLevel <nearestNeighbors) {
				isFind = false;
				for(int j=0;j<borderInst.size();j++){
					if (borderInst.get(j).instance(0).classValue() == prob.instance(i).classValue()) {
						borderInst.get(j).add(prob.instance(i));
						isFind = true;
						break;
					}
				}
				if (!isFind) {
					Instances temp = new Instances(prob, i, 1);
					borderInst.add(temp);//i-th sample is a borderline example
				}
			}
		}
			
		int maxInstance = 0;
		ArrayList<Instances> borderInstFinal = new ArrayList<>();
		for(int i=0;i<inst.size();i++){
			boolean isFind = false;
			for(int j=0;j<borderInst.size();j++){
				if (borderInst.get(j).instance(0).classValue() == inst.get(i).instance(0).classValue()) {
					borderInstFinal.add(borderInst.get(j));
					isFind = true;
					break;
				}
			}
			if (!isFind) {
				borderInstFinal.add(inst.get(i));
			}
			
			if (inst.get(i).numInstances() > maxInstance) {
				maxInstance = inst.get(i).numInstances();
			}
		}
		
		//generate instances
		Instance[] nnArray = new Instance[nearestNeighbors+2];
		for(int i=0;i<borderInstFinal.size();i++){
			//generate synthetic instances for i-th class
			if (maxInstance - inst.get(i).numInstances()<=0) {
				continue;
			}
			int nInstances = maxInstance - inst.get(i).numInstances();  //number needed for i-th class
			
			for (int j = 0; j < borderInstFinal.get(i).numInstances(); j++) {
				Instance instanceJ = borderInstFinal.get(i).instance(j);
				List distanceToInstance = new LinkedList();
				
				for(int p=0;p<inst.get(i).numInstances();p++){
					// find k nearest neighbors for each instance
					Instance instanceP = inst.get(i).instance(p);
					
					double distance = 0;
					for (int index=0;index<prob.numAttributes()-1;index++){
						double pValue = instanceP.value(index);
						double jValue = instanceJ.value(index);
						distance += Math.pow(pValue-jValue, 2);
					}
					distance = Math.pow(distance, 0.5);
					distanceToInstance.add(new Object[] { distance, instanceP });
				}
				
				//sort the instances according their distance to i-th sample
				Collections.sort(distanceToInstance, new Comparator() {
			        public int compare(Object o1, Object o2) {
			          double distance1 = (Double) ((Object[]) o1)[0];
			          double distance2 = (Double) ((Object[]) o2)[0];
			          return Double.compare(distance1, distance2);
			        }
			    });
					
				// populate the actual nearest neighbor instance array, store the nearest neighboor into nnArray
			    Iterator entryIterator = distanceToInstance.iterator();
			    int pos = 0;
			    while (entryIterator.hasNext() && pos <= nearestNeighbors+1) {
			    	nnArray[pos] = (Instance) ((Object[]) entryIterator.next())[1];
			    	pos++;
			    }
			    //compute the number of synthetic examples around i-th sample
			    int thisN = nInstances/borderInstFinal.get(i).numInstances();
			    if ((nInstances%borderInstFinal.get(i).numInstances())>j) {
					thisN++;
				}
			    
			    int target = (pos-1);
				    
			    while(thisN>0){
			    	double[] values = new double[prob.numAttributes()];
			        int nn = rand.nextInt(target)+1;          //randomly choosing a neighbor from i-th sample
			        
			        Enumeration attrEnum = prob.enumerateAttributes();
			        while (attrEnum.hasMoreElements()) {
			            Attribute attr = (Attribute) attrEnum.nextElement();
			            if (!attr.equals(prob.classAttribute())) {
			            	 double dif = nnArray[nn].value(attr) - instanceJ.value(attr);
			            	 double gap = rand.nextDouble();
			                 values[attr.index()] = (instanceJ.value(attr) + gap * dif);
			            }
			        }
			       
			        values[prob.classIndex()] = instanceJ.classValue();
			        Instance synthetic = new Instance(1.0, values);
			        artInstance.add(synthetic);                           //add the synthetic example into artInstance
			        thisN--;
			    }
			}
		}
		
		for (int i=0;i<prob.numInstances();i++){
			artInstance.add(prob.instance(i));
		}
		artInstance.setClassIndex(prob.numAttributes()-1);
		return artInstance;
	}
    
}
