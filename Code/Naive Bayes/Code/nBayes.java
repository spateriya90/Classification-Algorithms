/* 
 * CSE - 601 Fall 2017
 * Project 3 
 * 
 * Naive Bayes Classifier
 *  
 * 
 */

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class nBayes {
    static List < Integer > numAtt = new ArrayList < > ();
    static List < String > truthList = new ArrayList < > ();
    static List < String > truthList2 = new ArrayList < > ();
    private static DecimalFormat df = new DecimalFormat(".##");
    static double probAll1 = 0;
    static double probAll0 = 0;
    static int attNum = 0;
    static String t;
    static int mode = 0;
    static String testLine = "";
    static boolean flag = false;
    static Map < Integer, List < String >> dataMap = new HashMap < > ();
    static Map < Integer, List < String >> testDataMap = new HashMap < > ();
    static List < String > rawData;
    static List < String > rawTestData = new ArrayList < > ();
    static Map < Integer, List < Double >> data = new HashMap < > ();
    //	 static Map < Integer, List < Integer >> clustMap = new HashMap < > ();
    public static void main(String[] args) {
        // TODO Auto-generated method stub
    	
    	//Take input from user for Demo/Normal mode, input file name and Query
    	
        String inFile = "";
        int k = 10;
        Scanner s = new Scanner(System.in);
        System.out.println("Please Enter 1 for Demo and 2 for Normal Mode: ");
        mode = Integer.parseInt(s.nextLine());
        System.out.println("Please Enter Input File Name: ");
        inFile = s.nextLine();
        if (mode == 1) {
            System.out.println("Please Enter Test Query: ");
            testLine = s.nextLine();
        }


        getData(inFile);
    }

    public static void getData(String inFile) {

    	//For Normal Mode of operation
        if (mode == 2) {
        	//Read the file into rawData
            try {
                rawData = Files.readAllLines(Paths.get(inFile), StandardCharsets.UTF_8);
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            int all0 = 0;
            int all1 = 0;
            int rows = rawData.size();

            //Store data in map as <Row Number, List of Attributes for that row>
            int flag = 0;
            int index = 0;
            for (String s: rawData) {
                //		System.out.println(s);
                List < String > list = new ArrayList < > ();
                attNum = s.split("\\s+").length;
                int ind = 0;
                for (String s1: s.split("\\s+")) {
                    //			System.out.println("String s1 is "+s1);

                    if (ind != attNum - 1) {
                        list.add(s1);
                        if (flag == 0)
                            if (isNumeric(s1)) {
                                numAtt.add(ind);
                            }
                    } else {
                    	//Store actual labels into truthList
                        truthList.add(s1);
                        if (s1.equals("1"))
                            all1++;
                        else
                            all0++;
                    }
                    ind++;

                }
                //		break;
                flag = 1;
                dataMap.put(index++, list);
            }
        
            //Create Training and Testing data maps
            Map < Integer, List < String >> trMap = new HashMap < > ();
            Map < Integer, List < String >> teMap = new HashMap < > ();

            int sizeOfTest = rawData.size() / 10;
            int sizeOfTrain = rawData.size() - sizeOfTest;

            double accuracy = 0;
            List < List < Double >> resList = new ArrayList < > ();
            
            //Run 10-Fold Cross Validation
            for (int i = 0; i < 10; i++) {
                trMap.clear();
                teMap.clear();
                
                //Separate data into Testing and Training
                for (int j: dataMap.keySet()) {
                    if ((j >= (i * sizeOfTest)) && j < ((i + 1) * sizeOfTest))
                        teMap.put(j, dataMap.get(j));
                    else
                        trMap.put(j, dataMap.get(j));
                }

                //Run Naive Bayes Classifier
                //Training classifier on trMap and Testing on teMap

                List < Double > res = trainAndTest(trMap, teMap);
                resList.add(res);

            }

            double avgAccuracy = 0;
            double avgPrecision = 0;
            double avgRecall = 0;
            double avgFMeasure = 0;
            
            //Calculate average of performance metrics
            
            for (List < Double > i: resList) {

                avgAccuracy += i.get(0);
                avgPrecision += i.get(1);
                avgRecall += i.get(2);
                avgFMeasure += i.get(3);

            }
            avgAccuracy = avgAccuracy / resList.size();
            avgPrecision = avgPrecision / resList.size();
            avgRecall = avgRecall / resList.size();
            avgFMeasure = avgFMeasure / resList.size();

            System.out.println("Average Accuracy is " + df.format(avgAccuracy * 100) + "%");
            System.out.println("Average Precision is " + df.format(avgPrecision * 100) + "%");
            System.out.println("Average Recall is " + df.format(avgRecall * 100) + "%");
            //	System.out.println("Average F-Measure is "+df.format(avgFMeasure ));
            System.out.println("Average F-Measure is " + avgFMeasure);


        } else {
        	//For Demo Mode
        	//Read training data file and test query
            try {
                rawData = Files.readAllLines(Paths.get(inFile), StandardCharsets.UTF_8);
                //			rawTestData = Files.readAllLines(Paths.get(t), StandardCharsets.UTF_8);
                //			System.out.println(testLine);
                rawTestData.add(testLine);

            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            int all0 = 0;
            int all1 = 0;
            int rows = rawData.size();

            //Store training data as <Row Number, List of attributes for that row>
            int flag = 0;
            int index = 0;
            for (String s: rawData) {
                //		System.out.println(s);
                List < String > list = new ArrayList < > ();
                attNum = s.split("\\s+").length;
                int ind = 0;
                for (String s1: s.split("\\s+")) {
                    //			System.out.println("String s1 is "+s1);

                    if (ind != attNum - 1) {
                        list.add(s1);
                        if (flag == 0)
                            if (isNumeric(s1)) {
                                numAtt.add(ind);
                            }
                    } else {
                        truthList.add(s1);
                        if (s1.equals("1"))
                            all1++;
                        else
                            all0++;
                    }
                    ind++;

                }
                //		break;
                flag = 1;
                dataMap.put(index++, list);
            }

            int tIndex = 0;
            for (String s: rawTestData) {
                //		System.out.println(s);
                List < String > list = new ArrayList < > ();
                attNum = s.split("\\s+").length;
                int ind = 0;
                for (String s1: s.split("\\s+")) {
                    //			System.out.println("String s1 is "+s1);

                    //			if(ind!=attNum-1){
                    list.add(s1);
             
                    ind++;

                }
                //		break;
                //		flag = 1;
                testDataMap.put(tIndex++, list);
            }
    
            //Put training and testing data to their respective maps

            Map < Integer, List < String >> trMap = new HashMap < > ();
            Map < Integer, List < String >> teMap = new HashMap < > ();

     
            double accuracy = 0;
            List < List < Double >> resList = new ArrayList < > ();
            //	for(int i = 0;i<10;i++){
            trMap.clear();
            teMap.clear();
            for (int j: dataMap.keySet()) {
               
                trMap.put(j, dataMap.get(j));
            }

            for (int j: testDataMap.keySet()) {
              
                teMap.put(j, testDataMap.get(j));
            }


        
            List < Double > res = trainAndTest2(trMap, teMap);
            //		break;
            resList.add(res);

         
        }

    }

    public static List < Double > trainAndTest(Map < Integer, List < String >> train, Map < Integer, List < String >> test) {

    	//Initialize data structures for mean and SD for 0 and 1 labels
        Map < Integer, Double > meanMap1 = new HashMap < > ();
        Map < Integer, Double > sdMap1 = new HashMap < > ();
        Map < Integer, Double > meanMap0 = new HashMap < > ();
        Map < Integer, Double > sdMap0 = new HashMap < > ();
        Map < Integer, Map < String, Integer >> catMap0 = new HashMap < > ();
        Map < Integer, Map < String, Integer >> catMap1 = new HashMap < > ();
        int total1 = 0;
        int total0 = 0;
        double totalProb0 = 0;
        double totalProb1 = 0;
        for (int i: train.keySet()) {
            if (truthList.get(i).equals("1"))
                total1++;
            else
                total0++;

        }
        
        //Calculate Class Prior Probability P(Hi)
        totalProb0 = total0 / (double) train.size();
        totalProb1 = total1 / (double) train.size();

       //Loop over all the attributes one by one to calculate mean,SD if continuous 
       //and descriptor prior probability for categorial data
        for (int i = 0; i < attNum - 1; i++) {



            if (numAtt.contains(i)) {

                double sum0 = 0;
                double sum1 = 0;
                int count0 = 0;
                int count1 = 0;
                List < Double > val0 = new ArrayList < > ();
                List < Double > val1 = new ArrayList < > ();

                for (int j: train.keySet()) {

                    if (truthList.get(j).equals("1")) {
                        //					total1++;
                        count1++;
                        val1.add(Double.parseDouble(train.get(j).get(i)));
                        sum1 += Double.parseDouble(train.get(j).get(i));

                    } else {
                        //					total0++;
                        count0++;
                        val0.add(Double.parseDouble(train.get(j).get(i)));

                        sum0 += Double.parseDouble(train.get(j).get(i));

                    }



                }


                //Calculate average of 0 and 1 labels for attribute i
                double avg0 = sum0 / count0;
                double avg1 = sum1 / count1;
                meanMap1.put(i, avg1);
                meanMap0.put(i, avg0);
                double sdSum0 = 0;
                double sdSum1 = 0;
                
                //Calculate SD of 0 and 1 labels for attribute i
                for (double k: val0) {
                    sdSum0 += Math.pow(k - avg0, 2);
                }
                for (double k: val1) {
                    sdSum1 += Math.pow(k - avg1, 2);
                }

                double sd0 = Math.sqrt(sdSum0 / (val0.size() - 1));
                double sd1 = Math.sqrt(sdSum1 / (val1.size() - 1));
             
                sdMap0.put(i, sd0);
                sdMap1.put(i, sd1);



            } else {
            	
                // Categorical attribute
            	
            	//Get count for each label occuring with 0 and 1 respectively
                double sum0 = 0;
                double sum1 = 0;
                int count0 = 0;
                int count1 = 0;
                Map < String, Integer > catCount0 = new HashMap < > ();
                Map < String, Integer > catCount1 = new HashMap < > ();

                for (int j: train.keySet()) {
                    if (truthList.get(j).equals("1")) {
                        count1++;
                    
                        catCount1.put(train.get(j).get(i), catCount1.getOrDefault(train.get(j).get(i), 0) + 1);

                    } else {
                        count0++;
                        catCount0.put(train.get(j).get(i), catCount0.getOrDefault(train.get(j).get(i), 0) + 1);
				

                    }
                }
                catMap0.put(i, catCount0);
                catMap1.put(i, catCount1);

            }
      
        }

        
        //Start Testing Phase
        
        Map < Integer, Double > result = new HashMap < > ();

        for (int j: test.keySet()) {

            List < String > li = test.get(j);
           
            //Calculate Descriptor Prior Probabilities P(X|H0) and P(X|H1)
            
            double prob0 = 1;
            double prob1 = 1;

            for (int k = 0; k < attNum - 1; k++) {
            	
            	//Check if attribute is continuous, if yes then calculate probability using PDF
                if (numAtt.contains(k)) {
                    prob0 *= getPDF(meanMap0.get(k), sdMap0.get(k), Double.parseDouble(li.get(k)));
                    prob1 *= getPDF(meanMap1.get(k), sdMap1.get(k), Double.parseDouble(li.get(k)));


                } else {
                	//Attribute is categorical
                	
                	int count0 = 0;
                    int count1 = 0;
                    if (catMap0.get(k).containsKey(li.get(k)))
                        count0 = catMap0.get(k).get(li.get(k));
                    if (catMap1.get(k).containsKey(li.get(k)))
                        count1 = catMap1.get(k).get(li.get(k));
                   
                    double p0 = (double) count0 / total0;
                    double p1 = (double) count1 / total1;
                   
                    //Multiply prob0 and prob1 with probability of current attribute
                    prob0 *= p0;
                    prob1 *= p1;

                }


            }

            //Multiply P(X|Hi) with P(Hi)
            double res;
            prob0 = prob0 * totalProb0;
            prob1 = prob1 * totalProb1;
         

            if (prob1 > prob0) {
                res = 1.0;
            } else {
                res = 0.0;
            }

            result.put(j, res);
        }
        
        
        //Calculate TP,FP,TN and FN for calculating performance metrics
        int correct = 0;
        int incorrect = 0;
        int predPos = 0;
        int predNeg = 0;
        int actPos = 0;
        int actNeg = 0;
        int corrPos = 0;
        int corrNeg = 0;
        int falPos = 0;
        int falNeg = 0;
        List < Double > out = new ArrayList < > ();
        for (int i: result.keySet()) {
            double found = result.get(i);
            double actual = Double.parseDouble(truthList.get(i));
            //			System.out.println("Found : "+found + "  " + "Actual: "+actual);
            if (found == 1)
                predPos++;
            if (found == 0)
                predNeg++;
            if (actual == 1)
                actPos++;
            if (actual == 0)
                actNeg++;
            if (found == actual) {
                //				System.out.println("Equal");
                correct++;
                if (found == 1) {
                    corrPos++;
                } else {
                    corrNeg++;
                }
            } else {
                //				System.out.println("UnEqual");
                incorrect++;

                if (found == 1) {
                    falPos++;
                } else {
                    falNeg++;
                }

            }
        }


        double accuracy = (double)(corrPos + corrNeg) / (corrPos + corrNeg + falPos + falNeg);
        double precision = (double) corrPos / (corrPos + falPos);
        if (Double.isNaN(precision))
            precision = 0;
        double recall = (double) corrPos / (corrPos + falNeg);
        if (Double.isNaN(recall))
            recall = 0;
        double fmeasure = (double)(2 * recall * precision) / (recall + precision);
        if (Double.isNaN(fmeasure))
            fmeasure = 0;
       

        out.add(accuracy);
        out.add(precision);
        out.add(recall);
        out.add(fmeasure);

        //		
        //		System.out.println("Accuracy is "+accuracy);
        //		System.out.println("Precision is "+precision);
        //		System.out.println("Recall is "+recall);
        //		System.out.println("F-Measure is "+fmeasure);
        //
        //		System.out.println(correct + " Size is "+result.size());
        return out;
    }
    
    
    //Formula for Probability Density Function, takes in mean, SD and value of x
    
    public static double getPDF(double mean, double sd, double val) {

        double eNum = Math.pow(val - mean, 2);
        double eDenom = 2 * sd * sd;
        double eFactor = -1 * (eNum / eDenom);
        double e = Math.exp(eFactor);

        double num = e;
        double denom = sd * Math.sqrt(2 * Math.PI);

        return num / denom;


    }

//Function to check if attribute is Numeric or not
    public static boolean isNumeric(String str) {
        try {
            double d = Double.parseDouble(str);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }

    //ONLY FOR DEMO
    //Function for Training and Testing during Demo
    public static List < Double > trainAndTest2(Map < Integer, List < String >> train, Map < Integer, List < String >> test) {


        Map < Integer, Double > meanMap1 = new HashMap < > ();
        Map < Integer, Double > sdMap1 = new HashMap < > ();
        Map < Integer, Double > meanMap0 = new HashMap < > ();
        Map < Integer, Double > sdMap0 = new HashMap < > ();
        Map < Integer, Map < String, Integer >> catMap0 = new HashMap < > ();
        Map < Integer, Map < String, Integer >> catMap1 = new HashMap < > ();
        int total1 = 0;
        int total0 = 0;
        double totalProb0 = 0;
        double totalProb1 = 0;
        for (int i: train.keySet()) {
            if (truthList.get(i).equals("1"))
                total1++;
            else
                total0++;

        }
        totalProb0 = total0 / (double) train.size();
        totalProb1 = total1 / (double) train.size();

      

        for (int i = 0; i < attNum; i++) {



            if (numAtt.contains(i)) {

                double sum0 = 0;
                double sum1 = 0;
                int count0 = 0;
                int count1 = 0;
                List < Double > val0 = new ArrayList < > ();
                List < Double > val1 = new ArrayList < > ();

                for (int j: train.keySet()) {

                    if (truthList.get(j).equals("1")) {
                        //						total1++;
                        count1++;
                        val1.add(Double.parseDouble(train.get(j).get(i)));
                        sum1 += Double.parseDouble(train.get(j).get(i));

                    } else {
                        //						total0++;
                        count0++;
                        val0.add(Double.parseDouble(train.get(j).get(i)));

                        sum0 += Double.parseDouble(train.get(j).get(i));

                    }



                }


                double avg0 = sum0 / count0;
                double avg1 = sum1 / count1;
                meanMap1.put(i, avg1);
                meanMap0.put(i, avg0);
                double sdSum0 = 0;
                double sdSum1 = 0;

                for (double k: val0) {
                    sdSum0 += Math.pow(k - avg0, 2);
                }
                for (double k: val1) {
                    sdSum1 += Math.pow(k - avg1, 2);
                }

                double sd0 = Math.sqrt(sdSum0 / (val0.size() - 1));
                double sd1 = Math.sqrt(sdSum1 / (val1.size() - 1));


                sdMap0.put(i, sd0);
                sdMap1.put(i, sd1);

                // Continuous attribute, use PDF



            } else {
                // Categorical attribute,
                double sum0 = 0;
                double sum1 = 0;
                int count0 = 0;
                int count1 = 0;
                Map < String, Integer > catCount0 = new HashMap < > ();
                Map < String, Integer > catCount1 = new HashMap < > ();

                for (int j: train.keySet()) {
                    if (truthList.get(j).equals("1")) {
                        count1++;
                       

                        catCount1.put(train.get(j).get(i), catCount1.getOrDefault(train.get(j).get(i), 0) + 1);

                    } else {
                        count0++;
                        catCount0.put(train.get(j).get(i), catCount0.getOrDefault(train.get(j).get(i), 0) + 1);

                    }
                }
                catMap0.put(i, catCount0);
                catMap1.put(i, catCount1);

            }



        }
      

        Map < Integer, Double > result = new HashMap < > ();

        for (int j: test.keySet()) {

            List < String > li = test.get(j);


            double prob0 = 1;
            double prob1 = 1;
            double px = 1;
            for (int k = 0; k < attNum; k++) {
                if (numAtt.contains(k)) {
                    prob0 *= getPDF(meanMap0.get(k), sdMap0.get(k), Double.parseDouble(li.get(k)));
                    prob1 *= getPDF(meanMap1.get(k), sdMap1.get(k), Double.parseDouble(li.get(k)));


                } else {
                    int count0 = 0;
                    int count1 = 0;
                    if (catMap0.get(k).containsKey(li.get(k)))
                        count0 = catMap0.get(k).get(li.get(k));
                    if (catMap1.get(k).containsKey(li.get(k)))
                        count1 = catMap1.get(k).get(li.get(k));
                   
                    double p0 = (double) count0 / total0;
                    double p1 = (double) count1 / total1;

                    double pmul = (double)(count0 + count1) / (double)(total0 + total1);
                    px = pmul * px;
              
                    prob0 *= p0;
                    prob1 *= p1;

                }


            }

            //			System.out.println("Prob0 is "+prob0 + "  Prob1 is "+prob1);

            double res;
            prob0 = prob0 * totalProb0;
            prob1 = prob1 * totalProb1;
            //			System.out.println("TotalProb0 is "+totalProb0 + "  TotalProb1 is "+totalProb1);
            double finP1 = prob1 / px;
            double finP0 = prob0 / px;
            //			System.out.println("Prob0 is "+prob0 + "  Prob1 is "+prob1);
            //			System.out.println("p(H0/X) is "+prob0 + "  p(H1/X) is "+prob1);
            System.out.println("P(H0/X) is " + finP0);
            System.out.println("P(H1/X) is " + finP1);

            if (prob1 > prob0) {
                res = 1.0;
            } else {
                res = 0.0;
            }

            result.put(j, res);
            System.out.println("Predicted label is " + res);

        }


        
        List < Double > out = new ArrayList < > ();
        
        return out;
    }


}