package com.company;

import java.util.Random;

public class Config {
    public static double radius = 5;
    public static double nWeight = 0.05;//коэффицент обучения веса
    public static double nCenter = 0.05;//коэффицент обучения центров
    public static double nSigma = 0.05;//коэффицент обучения размера рад. баз. функции
    public static double nKMeans = 0.05;//коэффицент к-усреднения

    public static Random random = new Random(3);

    public static int windowSize = 7; // размер скользящего окна

    public static int countHiddenNeurons = 100;

    public static int trainingCount = 100;
    public static int trainingCentersCount = 1;

}
