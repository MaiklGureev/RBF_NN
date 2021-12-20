package com.company.nn;

import com.company.Config;

import java.io.Serializable;
import java.util.List;

public class Neuron implements Serializable {

    public double[] arrayX;//входные данные
    public double[] arrayC; //центры нейрона
    public double sigma;
    public double[] arraySigma;
    public double f;//значение функции


    public void kMeansCorrection() {
        for (int i = 0; i < arrayX.length; i++) {
            arrayC[i] = arrayC[i] + Config.nKMeans * (arrayX[i] - arrayC[i]);
        }
    }

    public void calcSigmaForNeighborNeurons(List<Neuron> neighborNeurons) {
        sigma = 0;
        int counter = 0;
        for (Neuron neuron : neighborNeurons) {
            if (counter >= Config.radius) {
                break;
            }
            for (int n = 0; n < arrayC.length; n++) {
                sigma += Math.pow((arrayC[n] - neuron.arrayC[n]), 2);
            }
            counter++;
        }
        sigma = sigma / Config.radius;
        sigma = Math.sqrt(sigma);

        arraySigma = new double[arrayC.length];
        for (int i = 0; i < arraySigma.length; i++) {
            arraySigma[i] = sigma;
        }

    }

    public double calcWeightForFindCenter() {
        double weight = 0;
        for (int i = 0; i < arrayX.length; i++) {
            weight += Math.pow(arrayX[i] - arrayC[i], 2);
        }
        weight = Math.sqrt(weight);
        return weight;
    }

    //гауссовская радиальная функция активации 4.11 и 4.12
    void calcGaussianRadFun() {
        double u = 0;
        double x;
        double c;
        double s;
        //4.12
        for (int k = 0; k < arrayX.length; k++) {
            x = arrayX[k];
            c = arrayC[k];
            s = arraySigma[k];
            u += (Math.pow((x - c), 2.0) / Math.pow(s, 2.0));
        }
        //4.11
        f = Math.exp(-u / 2.0);
    }
}
