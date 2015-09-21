import java.util.ArrayList;
import java.util.List;

/**
 * Created by Sutula on 18.09.15
 */
public class NeuralLayer {

    private final List<Neuron> neurons;
    private final int inputsAmount;

    public NeuralLayer(int neuronsAmount, int inputsAmount) {
        this.inputsAmount = inputsAmount;
        neurons = new ArrayList<>(neuronsAmount);

        for (int i = 0; i < neuronsAmount; i++) {
            neurons.add(new Neuron(inputsAmount));
        }
    }

    public List<Double> processThisLayer(List<Double> inputs) {
        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : neurons) {
            outputs.add(neuron.getNeuronOutput(inputs));
        }

        return outputs;
    }

    public void calculateOutputLayerErrors(List<Double> desiredOutput) {
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            neuron.calculateDifference(desiredOutput.get(i) - neuron.getLastOutput());
        }

    }

    public void calculateLayerErrors(List<Double> desiredOutput) {
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            neuron.calculateDifference(desiredOutput.get(i));
        }

    }

    public List<Double> getErrorVector() {
        List<Double> output = new ArrayList<>();

        for (int i = 0; i < inputsAmount; i++) {

            double sum = 0.0;
            for (int j = 0; j < neurons.size(); j++) {
                sum += neurons.get(j).getErrorCoefficientMult(i);
            }
            output.add(sum);
        }

        return output;
    }

    public void updateWeights(List<Double> prevLayerOutput) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).updateWeights(prevLayerOutput);
        }
    }

    public int getLayerSize() {
        return neurons.size();
    }

    public List<Double> getLastOutput() {

        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : neurons) {
            outputs.add(neuron.getLastOutput());
        }

        return outputs;
    }
}
