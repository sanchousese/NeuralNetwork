import java.util.List;

/**
 * Created by Sutula on 18.09.15
 */
public class NeuralNetwork {

    private NeuralLayer input;
    private NeuralLayer hidden;
    private NeuralLayer output;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        input = new NeuralLayer(inputSize, inputSize);
        hidden = new NeuralLayer(hiddenSize, inputSize);
        output = new NeuralLayer(outputSize, hiddenSize);
    }

    public List<Double> getResult(List<Double> inputValues) throws Exception {
        if (input.getLayerSize() != inputValues.size()) {
            throw new Exception("Wrong size of the input list");
        }

        List<Double> inputOutput = input.processThisLayer(inputValues);
        List<Double> hiddenOutput = hidden.processThisLayer(inputOutput);
        return output.processThisLayer(hiddenOutput);
    }

    public void teachNetwork(List<Double> inputValues, List<Double> desiredOutput) throws Exception {
        getResult(inputValues);
        output.calculateOutputLayerErrors(desiredOutput);
        output.updateWeights(hidden.getLastOutput());

        hidden.calculateLayerErrors(output.getErrorVector());
        hidden.updateWeights(input.getLastOutput());

        input.calculateLayerErrors(hidden.getErrorVector());
        input.updateWeights(inputValues);

    }
}
