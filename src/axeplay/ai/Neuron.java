package axeplay.ai;

import java.io.Serializable;
import java.util.Arrays;

class Neuron implements Serializable, Cloneable {

    public static final long serialVersionUID = 1L;

    double value;
    double[] weights;
    Neuron(double value) {
        this.value = value;
        weights = new double[0];
    }

    @Override
    public Neuron clone() {
        Neuron result = new Neuron(1);
        result.weights = Arrays.copyOf(this.weights, this.weights.length);
        return result;
    }

}
