package axeplay.ai;

import java.io.*;

@SuppressWarnings("all")
public class NeuralNetwork implements Serializable {

    public static final long serialVersionUID = 2L;

    //f(x) = (1+e^-x)^(-1)
    //f'(x) = e^-x / (1+e^-x)^(2) = ((1 - f(x)) * f(x)

    private Neuron[][] nn;
    private double speedLearning;

    private double moment;
    private double[][][] deltaWeights;

    private Func activation = (Func & Serializable) x -> 1 / (1 + Math.exp(-x));
    private Func derivative = (Func & Serializable) y -> y * (1 - y);

    public NeuralNetwork(double speedLearning, double moment, int... layers) {
        this.moment = moment;
        this.speedLearning = speedLearning;
        nn = new Neuron[layers.length][];
        for (int l = 0; l < nn.length; l++) {
            nn[l] = new Neuron[layers[l] + ((l != nn.length-1) ? 1 : 0)];
            for (int n = 0; n < nn[l].length; n++) {
                nn[l][n] = new Neuron(1);
                if (l != 0 && (n != nn[l].length-1 || l == nn.length-1)) {
                    nn[l][n].weights = new double[nn[l - 1].length];
                    for (int w = 0; w < nn[l][n].weights.length; w++) {
                        nn[l][n].weights[w] = Math.random() * 2 - 1;
                    }
                }
            }
        }

        deltaWeights = new double[nn.length][][];
        for (int l = 0; l < deltaWeights.length-1; l++) {
            deltaWeights[l] = new double[nn[l].length][];
            for (int n = 0; n < deltaWeights[l].length; n++) {
                deltaWeights[l][n] = new double[nn[l+1].length];
            }
        }
    }

    public double[] apply(double... firstLayer) {
        if (firstLayer.length != nn[0].length-1)
            throw new RuntimeException("Количество входных данных не совпадает с количеством входных нейронов!");

        for (int n = 0; n < firstLayer.length; n++) {
            nn[0][n].value = firstLayer[n];
        }

        for (int l = 0; l < nn.length-1; l++) {
            Neuron[] l0 = nn[l];
            Neuron[] l1 = nn[l + 1];
            for (int n1 = 0, length = l1.length - (((l+1) != nn.length-1) ? 1 : 0); n1 < length; n1++) {
                l1[n1].value = 0;
                for (int n = 0; n < l0.length; n++) {
                    l1[n1].value += l0[n].value * l1[n1].weights[n];
                }
                l1[n1].value = activation.apply(l1[n1].value);
            }
        }

        double[] results = new double[nn[nn.length-1].length];
        for (int n = 0; n < results.length; n++) {
            results[n] = nn[nn.length-1][n].value;
        }
        return results;
    }

    public void backpropagation(double... results) {
        double[] errors = new double[results.length];
        for (int i = 0; i < errors.length; i++) {
            errors[i] = (results[i] - nn[nn.length-1][i].value);
        }

        double[] deltas = new double[errors.length];
        for (int i = 0; i < deltas.length; i++) {
            deltas[i] = errors[i] * derivative.apply(nn[nn.length-1][i].value);
        }

        for (int l = nn.length-2; l >= 0; l--) {
            Neuron[] l0 = nn[l];
            Neuron[] l1 = nn[l + 1];
            double[] newDeltas = new double[l0.length];
            for (int n = 0; n < l0.length; n++) {
                double sumWeights = 0;
                for (int w = 0; w < deltas.length - ((l != nn.length-2) ? 1 : 0); w++) {
                    sumWeights += l1[w].weights[n] * deltas[w];

                    double gradient = deltas[w] * l0[n].value;
                    deltaWeights[l][n][w] = speedLearning * gradient + moment * deltaWeights[l][n][w];
                    l1[w].weights[n] += deltaWeights[l][n][w];
                }
                newDeltas[n] = derivative.apply(l0[n].value) * sumWeights;
            }
            //deltas& = ^newDeltas;
            deltas = newDeltas;
        }
    }

    public NeuralNetwork clone() {
        NeuralNetwork result = new NeuralNetwork(0,0, 1);
        result.activation = this.activation;
        result.derivative = this.derivative;

        result.moment = this.moment;
        result.speedLearning = this.speedLearning;

        result.deltaWeights = this.deltaWeights.clone();

        result.nn = new Neuron[this.nn.length][];
        for (int l = 0; l < this.nn.length; l++) {
            result.nn[l] = new Neuron[this.nn[l].length];
            for (int n = 0; n < this.nn[l].length; n++) {
                result.nn[l][n] = this.nn[l][n].clone();
            }
        }
        return result;
    }

    public void setActivationFunction(Func activation) {
        this.activation = activation;
    }

    public void setDerivativeFunction(Func derivative) {
        this.derivative = derivative;
    }

    public void setSpeedLearning(double speedLearning) {
        this.speedLearning = speedLearning;
    }

    public void setMoment(double moment) {
        this.moment = moment;
    }

    @Override
    public String toString() {
        String result = "SpeedLearning: " + speedLearning + " Moment: " + moment + "\n";
        for (int l = 0; l < nn.length; l++) {
            result += "Layer: " + l + "\n";
            for (int n = 0; n < nn[l].length; n++) {
                result += "Neuron: " + n + " Value: " + nn[l][n].value;
                for (int w = 0; w < nn[l][n].weights.length; w++) {
                    result += " Weight: " + w + " " + nn[l][n].weights[w];
                }
                result += "\n";
            }
            result += "\n";
        }
        return result;
    }

    public void save(String path) throws IOException {
        FileOutputStream outputStream = new FileOutputStream(path);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(outputStream);

        objectOutputStream.writeObject(this);
        objectOutputStream.close();
    }

    public static NeuralNetwork load(String path) throws IOException, ClassNotFoundException {
        FileInputStream fileInputStream = new FileInputStream(path);
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);

        return (NeuralNetwork) objectInputStream.readObject();
    }

}
