package com.company.nn;

import com.company.Config;
import com.company.dataset.Dataset;
import com.company.dataset.Dataset2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork implements Serializable {

    public List<Neuron> neurons;
    public OutNeuron outNeuron;

    public double[] datasetTrain = Dataset2.train;
    public double[] datasetTest = Dataset2.test;

    public NeuralNetwork() {
        neurons = new ArrayList<Neuron>();

        for (int i = 0; i < Config.countHiddenNeurons; i++) {
            neurons.add(new Neuron());
        }

        outNeuron = new OutNeuron(Config.countHiddenNeurons);

        initCenters();
    }


    public void kMeans() {

        int startIndex = 0;
        int currentIndex;
        int countArrays = (int) Math.floor(datasetTrain.length / Config.windowSize);
        double[] arrayX;

        for (int a = 0; a < countArrays; a++) {
            currentIndex = startIndex + Config.windowSize;

            arrayX = getArrayX(datasetTrain, startIndex, currentIndex);

            //System.out.println(Arrays.toString(arrayX));

            int winningCenterId = 0;
            double weight = 0;
            double[] weightsArray = new double[neurons.size()];

            //цикл по нейронам
            for (int n = 0; n < neurons.size(); n++) {
                //подсчёт весов в нейроне
                neurons.get(n).arrayX = arrayX;
                weight = neurons.get(n).calcWeightForFindCenter();
                weightsArray[n] = weight;
            }

            //поиск id нейрона с минимальным весом
            winningCenterId = findWinningCenterId(weightsArray);
            //коррекция центров у нерона
            neurons.get(winningCenterId).kMeansCorrection();

            startIndex = currentIndex;
        }

    }

    public void trainAndTest() {

        for (int t = 0; t < Config.trainingCentersCount; t++) {
            kMeans();
        }

        calcSigmaInHideNeurons();
        train();
        System.out.println("--------------------------------------------------------------------------------------------------------------------------------------------------");
        test();
    }

    private void train() {

        System.out.println("--------------------------------------------------------------------------------------------------------------------------------------------------");

        for (int t = 0; t < Config.trainingCount; t++) {

            int currentIndex = Config.windowSize;
            double[] arrayX = getArrayX(datasetTrain, 0, currentIndex);
            double[] forecastingArray = new double[datasetTrain.length];

            for (int i = 0; i < Config.windowSize; i++) {
                forecastingArray[i] = datasetTrain[i];
            }

            outNeuron.mse = 0;

            while (currentIndex != datasetTrain.length) {
                //цикл по нейронам
                for (int n = 0; n < neurons.size(); n++) {
                    neurons.get(n).arrayX = arrayX;
                }
                outNeuron.d = datasetTrain[currentIndex];

                calcGaussianRadFunctionsForHideNeurons();

                recalculateWeightsForNeurons(outNeuron);
                recalculateCentersAndSigmaForNeurons(outNeuron);

                outNeuron.calcLocalMSE();

                forecastingArray[currentIndex] = outNeuron.calcMainFun(neurons);

                arrayX = addNextValueToArray(arrayX, datasetTrain[currentIndex]);

                currentIndex++;
            }
            System.out.println("datasetTrain = " + Arrays.toString(datasetTrain));
            System.out.println("forecastingArray = " + Arrays.toString(forecastingArray));
            System.out.println(t + " MSE = " + outNeuron.calcMSE(datasetTrain.length - 1));
        }

    }


    public void test() {
        int currentIndex = 0;
        double[] arrayX = getLastElementsInArray(datasetTrain, Config.windowSize);
        double[] forecastingArray = new double[datasetTest.length];
        outNeuron.d = datasetTest[0];
        outNeuron.mse = 0;

        while (currentIndex != datasetTest.length) {

            //цикл по нейронам
            for (int n = 0; n < neurons.size(); n++) {
                neurons.get(n).arrayX = arrayX;
            }

            outNeuron.d = datasetTest[currentIndex];
            calcGaussianRadFunctionsForHideNeurons();
            forecastingArray[currentIndex] = outNeuron.calcMainFun(neurons);

            System.out.println(currentIndex + " error = " + outNeuron.calcLocalMSE());

            arrayX = addNextValueToArray(arrayX, forecastingArray[currentIndex]);

            currentIndex++;
        }
        System.out.println("------------------------------------------------------------------------------------------------------------------------------------------------");
        System.out.println("MSE = " + outNeuron.calcMSE(datasetTest.length - 1));

        System.out.println("datasetTest = " + Arrays.toString(datasetTest));
        System.out.println("forecastingArray = " + Arrays.toString(forecastingArray));
        System.out.println(formatArrayForPrint(forecastingArray));

    }

    private String formatArrayForPrint(double[] forecastingArray) {
        double[] temp = new double[forecastingArray.length];
        for (int i = 0; i < forecastingArray.length; i++) {
            temp[i] = forecastingArray[forecastingArray.length - 1 - i];
        }
        String s = Arrays.toString(forecastingArray);
        s = s.replace(",", "\n");
        s = s.replace(".", ",");
        s = s.replace("[", "");
        s = s.replace("]", "");
        return s;
    }


    //пересчёт весов у нейронов
    public void recalculateWeightsForNeurons(OutNeuron outNeuron) {
        int i;
        double p;

        for (Neuron neuron : neurons) {
            i = neurons.indexOf(neuron);
            p = neuron.f * (outNeuron.y - outNeuron.d);
            outNeuron.w[i] = outNeuron.w[i] - Config.nWeight * p;
        }

        p = outNeuron.y - outNeuron.d;
        outNeuron.w0 = outNeuron.w0 - Config.nWeight * p;
    }

    //пересчёт центров и сигмы у нейронов
    public void recalculateCentersAndSigmaForNeurons(OutNeuron outNeuron) {
        double p;
        double sum;
        int currentIndex;

        for (Neuron neuron : neurons) {
            currentIndex = neurons.indexOf(neuron);
            sum = 0;

            sum += (outNeuron.y - outNeuron.d) * outNeuron.w[currentIndex];

            // System.out.println("neuron 0 arrayC = " + Arrays.toString(neuron.arrayC));
            // System.out.println("neuron 0 arraySigma = " + Arrays.toString(neuron.arraySigma));

            for (int j = 0; j < neuron.arrayX.length; j++) {

                //4.15a
                p = sum * neuron.f * ((neuron.arrayX[j] - neuron.arrayC[j]) / Math.pow(neuron.arraySigma[j], 2));
                neuron.arrayC[j] = neuron.arrayC[j] - Config.nCenter * p;

                //4.16a
                p = sum * neuron.f * ((neuron.arrayX[j] - neuron.arrayC[j]) / Math.pow(neuron.arraySigma[j], 3));
                neuron.arraySigma[j] = neuron.arraySigma[j] - Config.nSigma * p;

            }

            // System.out.println("neuron 0 newArrayC = " + Arrays.toString(neuron.arrayC));
            // System.out.println("neuron 0 newArraySigma = " + Arrays.toString(neuron.arraySigma));

        }

    }

    public int findWinningCenterId(double[] weight) {
        double minWeight = weight[0];
        int id = 0;
        for (int i = 0; i < weight.length; i++) {
            if (weight[i] < minWeight) {
                id = i;
            }
        }
        return id;
    }

    public void calcSigmaInHideNeurons() {
        List<Neuron> neuronsNeighbors = new ArrayList<>();
        for (Neuron n : neurons) {
            neuronsNeighbors.addAll(neurons);
            neuronsNeighbors.remove(n);
            n.calcSigmaForNeighborNeurons(neuronsNeighbors);
            neuronsNeighbors = new ArrayList<>();
        }
    }

    //расчёт значений функций для нейронов
    void calcGaussianRadFunctionsForHideNeurons() {
        for (Neuron neuron : neurons) {
            neuron.calcGaussianRadFun();
        }
    }

    private double[] getArrayX(double[] input, int startIndex, int endIndex) {
        double[] arrayX = new double[Config.windowSize];
        int i = 0;
        for (int j = startIndex; j < endIndex; j++) {
            arrayX[i] = input[j];
            i++;
        }
        return arrayX;
    }

    double[] addNextValueToArray(double[] currentArray, double nextValue) {
        double[] tempArray = new double[currentArray.length];
        for (int i = 0; i < currentArray.length - 1; i++) {
            tempArray[i] = currentArray[i + 1];
        }
        tempArray[tempArray.length - 1] = nextValue;
        return tempArray;
    }

    private double[] getLastElementsInArray(double[] input, int windowSize) {
        double[] tempArray = new double[windowSize];
        int startIndex = input.length - windowSize;
        int currentIndex = 0;
        for (int i = startIndex; i < input.length; i++) {
            tempArray[currentIndex] = input[i];
            currentIndex++;
        }
        return tempArray;
    }

    private void initCenters() {

        for (Neuron neuron : neurons) {

            int randomIndexInTrainSet = Config.random.nextInt(datasetTrain.length);

            if (randomIndexInTrainSet + Config.windowSize > datasetTrain.length) {
                randomIndexInTrainSet = datasetTrain.length - Config.windowSize - 1;
            }

            neuron.arrayC = getArrayX(
                    datasetTrain,
                    randomIndexInTrainSet,
                    randomIndexInTrainSet + Config.windowSize
            );

        }
    }


    @Override
    public String toString() {
        return "NeuralNetwork{" +
                ", neurons.size=" + neurons.size() +
//                ", neurons=" + neurons +
//                ", outNeuron=" + outNeuron +
                '}';
    }
}
