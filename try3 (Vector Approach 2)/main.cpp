#include <cstdlib>
#include <iostream>
#include <vector>

struct Connection {
  double weight;
  double deltaWeight;
};
class Neuron;

typedef std::vector<Neuron> Layer;

// ********************** Class Neuron **************************

class Neuron {
public:
  Neuron(unsigned numOutputs);

private:
  static double randomWeight() { return rand() / double(RAND_MAX); }
  double m_outputVal;
  std::vector<Connection> m_outputWeights;
};
Neuron::Neuron(unsigned numOutputs) {
  for (unsigned c = 0; c < numOutputs; c++) {
    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight = randomWeight();
  }
};

// ********************** Class Net **************************
class Net {
public:
  Net(const std::vector<unsigned> &topology);
  void feedforward(std::vector<double> &inputVals) {};
  void backProp(const std::vector<double> &targetVals) {};
  void getResults(const std::vector<double> resultVals) const {};

private:
  std::vector<Layer> m_layers;
};

Net::Net(const std::vector<unsigned> &topology) {
  unsigned numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
    m_layers.push_back(Layer());
    unsigned numOutputs =
        layerNum == numLayers - 1 ? 0 : topology[layerNum + 1];

    for (unsigned neuronNumber = 0; neuronNumber <= topology[layerNum];
         neuronNumber++) {
      m_layers.back().push_back(Neuron(numOutputs));
      std::cout << "Made a new Neuron!" << std::endl;
    }
  }
}
int main(int argc, char *argv[]) {
  std::vector<unsigned> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(1);
  Net myNet(topology);

  std::vector<double> inputVals;
  myNet.feedforward(inputVals);

  std::vector<double> targetVals;
  myNet.backProp(targetVals);

  std::vector<double> resultVals;
  myNet.getResults(resultVals);

  return 0;
}
