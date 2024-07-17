import axeplay.ai.NeuralNetwork;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        digits();
    }

    private static void digits() throws IOException {
        NeuralNetwork nn;
        try {
            nn = NeuralNetwork.load(".\\save.nn");
        } catch (Exception e) {
            nn = new NeuralNetwork(0.01, 0.4, 784, 512, 128, 32, 10);

            int samples = 60000;
            BufferedImage[] images = new BufferedImage[samples];
            int[] digits = new int[samples];
            File[] imagesFiles = new File("./train").listFiles();
            for (int i = 0; i < samples; i++) {
                images[i] = ImageIO.read(imagesFiles[i]);
                digits[i] = Integer.parseInt(imagesFiles[i].getName().charAt(10) + "");
            }

            double[][] inputs = new double[samples][784];
            for (int i = 0; i < samples; i++) {
                for (int x = 0; x < 28; x++) {
                    for (int y = 0; y < 28; y++) {
                        inputs[i][x + y * 28] = (images[i].getRGB(x, y) & 0xff) / 255.0;
                    }
                }
            }

            nn.setMoment(0.4);
            nn.setSpeedLearning(0.01);
            int epochs = 600;
            for (int i = 0; i < epochs; i++) {
                if (i / 200 == 1) {
                    nn.setMoment(0.2);
                    nn.setSpeedLearning(0.05);
                }
                if (i / 200 == 2) {
                    nn.setMoment(0.05);
                    nn.setSpeedLearning(0.025);
                }
                int right = 0;
                double errorSum = 0;
                int batchSize = 100;
                for (int j = 0; j < batchSize; j++) {
                    int imgIndex = i * 100 + j;
                    double[] targets = new double[10];
                    int digit = digits[imgIndex];
                    targets[digit] = 1;

                    double[] outputs = nn.apply(inputs[imgIndex]);
                    int maxDigit = 0;
                    double maxDigitWeight = -1;
                    for (int k = 0; k < 10; k++) {
                        if (outputs[k] > maxDigitWeight) {
                            maxDigitWeight = outputs[k];
                            maxDigit = k;
                        }
                    }
                    if (digit == maxDigit) right++;
                    for (int k = 0; k < 10; k++) {
                        errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                    }
                    nn.backpropagation(targets);
                }
                System.out.println("epoch: " + i + ". correct: " + right + ". error: " + errorSum);
            }

            nn.save(".\\save.nn");
        }

        FormDigits f = new FormDigits(nn);
        new Thread(f).start();
    }

}