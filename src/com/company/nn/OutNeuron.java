package com.company.nn;

import com.company.Config;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class OutNeuron implements Serializable {

    public double w0;
    public double y = 0;
    public double[] w;//веса
    public double d; //ожидаймое зачение
    public double mse = 0;

    public OutNeuron(int countHiddenNeurons) {
        w = new double[countHiddenNeurons];
        //инициализация весов
        w0 = 0.5 * Config.random.nextDouble();
        for (int i = 0; i < countHiddenNeurons; i++) {
            w[i] = 0.5 * Config.random.nextDouble();
        }
    }

    double calcMainFun(List<Neuron> neurons) {
        y = 0;
        y += w0 * 1;
        int i = 0;
        for (Neuron neuron : neurons) {
            y += w[i] * neuron.f;
            i++;
        }
        return y;
    }

    //функция MSE
    double calcMSE(double m) {
        mse = Math.sqrt(mse * (1.0 / (m - 1.0)));
        return mse;
    }

    double calcLocalMSE() {
        double localMSE = Math.pow((y - d), 2);
        mse += localMSE;
        return Math.sqrt(localMSE);
    }

    @Override
    public String toString() {
        return "\n\t\tOutNeuron{" +
                ", w0=" + w0 +
                ", w=" + Arrays.toString(w) +
                '}';
    }
}
