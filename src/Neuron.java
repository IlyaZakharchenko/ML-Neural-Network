import java.util.ArrayList;
import java.util.List;

class Neuron {

    private List<Double> weights;
    private List<Double> inputData;
    private Double output = 0.0;
    private Double delta;
    private double threshold = 0.5 - Math.random();

    Neuron(int numberOfInputs) {
        weights = new ArrayList<>();
        for (int i = 0; i < numberOfInputs; i++) {
            weights.add(0.5 - Math.random());
        }
    }

    double getThreshold() {
        return threshold;
    }

    void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    void activate(double result) {
        output = 1 / (1 + Math.exp(-result));
    }

    Double derivation() {
        return output * (1 - output);
    }

    Double getDelta() {
        return delta;
    }

    void setDelta(Double coef) {
        delta = derivation() * coef;
    }

    List<Double> getWeights() {
        return weights;
    }

    List<Double> getInputData() {
        return inputData;
    }

    void setInputData(List<Double> inputData) {
        this.inputData = inputData;
    }

    Double getOutput() {
        return output;
    }
}
