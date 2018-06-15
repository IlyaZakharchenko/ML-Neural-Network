import java.util.ArrayList;
import java.util.List;

public class MNISTNeuralNetwork {
    private final int EPOCH_NUM = 1000000;
    private final int BACH_NUM = 4;
    private final int HIDDEN_LAYER_SIZE = 8;
    private final int INPUT_LAYER_SIZE = 2;
    private final int OUTPUT_LAYER_SIZE = 1;
    private final Double LEARNING_RATE = 1.0;
    private List<List<Double>> data;
    private List<Double> desiredData;
    private List<Neuron> neurons;

    private void initHiddenLayer() {
        neurons = new ArrayList<>();
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            neurons.add(new Neuron(INPUT_LAYER_SIZE));
        }
    }

    private void initOutputLayer() {
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            neurons.add(new Neuron(HIDDEN_LAYER_SIZE));
        }
    }

    private void initData() {
        data = new ArrayList<>();
        List<Double> data1 = new ArrayList<>();
        data1.add(0.0);
        data1.add(0.0);
        data.add(data1);
        data1 = new ArrayList<>();
        data1.add(0.0);
        data1.add(1.0);
        data.add(data1);
        data1 = new ArrayList<>();
        data1.add(1.0);
        data1.add(0.0);
        data.add(data1);
        data1 = new ArrayList<>();
        data1.add(1.0);
        data1.add(1.0);
        data.add(data1);

        desiredData = new ArrayList<>();
        desiredData.add(0.0);
        desiredData.add(1.0);
        desiredData.add(1.0);
        desiredData.add(0.0);
    }

    private double propagateForward(List<Double> inputData) {
        double weightedSum = 0;
        //hidden layer
        List<Double> nextLayerData = new ArrayList<>();
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            Neuron neuron = neurons.get(i);
            neuron.setInputData(inputData);
            List<Double> weights = neuron.getWeights();
            List<Double> input = neuron.getInputData();
            for (int j = 0; j < weights.size(); j++) {
                weightedSum += neuron.getThreshold() + weights.get(j) * input.get(j);
            }
            neuron.activate(weightedSum);
            nextLayerData.add(neuron.getOutput());
        }
        //output layer
        for (int i = HIDDEN_LAYER_SIZE; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            neuron.setInputData(nextLayerData);
            List<Double> weights = neuron.getWeights();
            List<Double> input = neuron.getInputData();
            for (int j = 0; j < weights.size(); j++) {
                weightedSum += neuron.getThreshold() + weights.get(j) * input.get(j);
            }
            neuron.activate(weightedSum);
        }
        return neurons.get(neurons.size() - 1).getOutput();
    }

    private void propagateBackward(double desiredResult) {
        double[] deltaWeightSums = new double[neurons.get(neurons.size() - 1).getWeights().size()];
        for (int i = HIDDEN_LAYER_SIZE; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            neuron.setDelta((desiredResult - neuron.getOutput()) * neuron.derivation());
            for (int j = 0; j < neuron.getWeights().size(); j++) {
                deltaWeightSums[j] += neuron.getWeights().get(j) * neuron.getDelta();
            }
        }

        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            Neuron neuron = neurons.get(i);
            neuron.setDelta(deltaWeightSums[i] * neuron.derivation());
        }

        for (Neuron neuron : neurons) {
            for (int i = 0; i < neuron.getWeights().size(); i++) {
                neuron.setThreshold(neuron.getThreshold() * neuron.getDelta());
                neuron.getWeights().set(i, neuron.getWeights().get(i) + LEARNING_RATE * neuron.getDelta() * neuron.getInputData().get(i));
            }
        }
    }

    void run() {
        initData();
        initHiddenLayer();
        initOutputLayer();
        for (int i = 0; i < EPOCH_NUM; i++) {
            Double cost = 0.0;
            for (int j = 0; j < BACH_NUM; j++) {
                Double result;
                result = propagateForward(data.get(j));
                propagateBackward(desiredData.get(j));
                cost = desiredData.get(j) - result;
            }
            System.out.println("Epoch = " + i + " Cost = " + cost);
        }
        predict();
    }

    private void predict() {
        for (int j = 0; j < BACH_NUM; j++){
            System.out.println(propagateForward(data.get(j)) + "  " + desiredData.get(j));
        }
    }
}
