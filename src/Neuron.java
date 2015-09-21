import java.util.ArrayList;
import java.util.List;

/**
 * Created by Sutula on 18.09.15
 */
public class Neuron {

    private static final double MAX_RANDOM = 0.39;
    private static final double LEARNING_RATE = 0.25;
    private double threshold;


    private final List<Double> weights;
    private double lastOutput;

    private double errorValue;

    public Neuron(int numberOfNeuronsInPrevLayer) {
        weights = new ArrayList<>(numberOfNeuronsInPrevLayer);

        for (int i = 0; i < numberOfNeuronsInPrevLayer; i++) {
            weights.add(Math.random() * MAX_RANDOM + 0.1);
        }

        threshold = Math.random() * MAX_RANDOM + 0.1;
    }

    public Double getNeuronOutput(List<Double> inputs) {
        double activity = 0;
        for (int i = 0; i < inputs.size(); i++) {
            activity += inputs.get(i) * weights.get(i);
        }
        lastOutput = getOutputValue(activity);
        return lastOutput;
    }

    public void updateWeights(List<Double> prevLayerOutput) {
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) + LEARNING_RATE * errorValue * prevLayerOutput.get(i));
        }

        threshold -= LEARNING_RATE * errorValue;
    }

    // TODO: use strategy
    private double getOutputValue(double neuronActivity) {
        return 1 / (1 + Math.exp(-(neuronActivity - threshold)));
    }

    public double getLastOutput() {
        return lastOutput;
    }

    public void calculateDifference(Double difference) {
        errorValue = difference * lastOutput * (1 - lastOutput);
    }

    public double getErrorCoefficientMult(int index) {
        return errorValue * weights.get(index);
    }

    @Override
    public String toString() {
        return "Neuron{" +
                "weights=" + weights +
                ", errorValue=" + errorValue +
                '}';
    }
}
