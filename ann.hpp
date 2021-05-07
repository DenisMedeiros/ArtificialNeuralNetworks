#ifndef ANN_HPP_
#define ANN_HPP_

#include <vector>
#include <memory>

/** Single-Layer Perceptron */
class SLP
{
    private:
        std::vector<double> weights;
        std::vector<std::vector<double>> inputs;
        std::vector<double> desired;
        bool debug;
        
        // Activation functions.
        double sigm(double);
        double dsigm(double);
        double sign(double);
        double step(double);
    public:
        SLP(unsigned int, bool _debug = false);
        void train(std::vector<std::vector<double>> &, std::vector<double> &, double, double, unsigned int);
        double process(std::vector<double> &);
        void printWeights(void) const;

};

#endif /* ANN_HPP_ */
