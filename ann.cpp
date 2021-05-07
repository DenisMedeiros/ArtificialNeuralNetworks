#include "ann.hpp"

#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <vector>
#include <stdexcept>

SLP::SLP(unsigned int numInputs, bool _debug)
{
	// Ensure number of inputs is greater than zero.
	if (numInputs < 1)
	{
		throw std::invalid_argument("Number of inputs must be greater than 1.");
	}
        
	// Allocate space for weights vector.
	weights = std::vector<double>(numInputs + 1);

	// Define PRNG.
	std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Define lambda function to generate random values.
    auto generator = [&dist, &engine](){ return dist(engine);};

    // Fill weights vector with random values.
    std::generate(weights.begin(), weights.end(), generator);
    
    debug = _debug;
}


void SLP::printWeights() const
{
	for(auto w: weights)
	{
		std::cout << w << std::endl;
	}
}

void SLP::train(std::vector<std::vector<double>> &inputs, std::vector<double> &desired, double learningRate, double stopMSE, unsigned int stopIter)
{
	std::vector<double> current(desired.size());
	std::vector<double> error(desired.size());
	double mse;

	// Ensure inputs and desired outputs are not empty.
	if (inputs.empty() || desired.empty())
	{
		throw std::invalid_argument("Inputs or desired outputs can't be empty.");
	}

	// Ensure inputs and desired outputs must have same size.
	if (inputs.size() != desired.size())
	{
		throw std::logic_error("Inputs or desired outputs must have same size.");
	}

	// Ensure inputs has expected number of entries (as defined in weights).
	if (inputs[0].size() != weights.size() - 1)
	{
		throw std::logic_error("Inputs must follow the defined number of inputs.");
	}


	for (auto i = 0; i < stopIter; i++) // Iterations.
	{
		// Calculate current outputs.
		for (auto j = 0; j < inputs.size(); j++) // itera nas linhas (0,0), 0,1).
		{
			current[j] = -1.0 * weights[0];
			
			for(auto k = 0; k < inputs[0].size(); k++) // itera nos elementos.
			{
				current[j] += inputs[j][k] * weights[k+1];
			}

			current[j] = step(current[j]);
		}

		// Calculate error.
		for (auto j = 0; j < error.size(); j++)
		{
			error[j] = desired[j] - current[j];
		}

		// Update weights.
		double inc = 0.0;
		for (auto j = 0; j < error.size(); j++)
		{
			inc += -1.0 * error[j];
		}
		weights[0] += learningRate * inc;

		for (auto j = 1; j < weights.size(); j++)
		{
			inc = 0.0;

			for (auto k = 0; k < inputs.size(); k++) // itera nas linhas (0,0), 0,1).
			{
				for(auto l = 0; l < inputs[0].size(); l++) // itera nos elementos.
				{
					inc += error[k] * inputs[k][l];
				}
			}

			weights[j] += learningRate * inc;
		}

		// Calculate MSE.
		mse = (1.0/error.size()) * std::inner_product(error.begin(), error.end(), error.begin(), 0);

		if(debug)
		{
			std::cout << "[debug] " << "Iteration = " << i << ", MSE = " << mse << "." << std::endl;
		}

		if(mse < stopMSE)
		{
			break;
		}

	}
	
	if(debug)
	{
		std::cout << "[debug] " << "Weights: ";
		for (auto w: weights)
		{
			std::cout << w << " ";
		}
		std::cout << std::endl;
	}

}

double SLP::process(std::vector<double> &_input)
{
	double output;

	output = -1.0 * weights[0];
	for(auto i = 0; i < _input.size(); i++)
	{
		output += _input[i] * weights[i+1];
	}

	return step(output);
}

double SLP::sigm(double x)
{
	return 1.0/(1.0 + exp(-x));
}

double SLP::dsigm(double x)
{
	return this->sigm(x)*(1 - this->sigm(x));
}

double SLP::sign(double x)
{
	return (x > 0) - (x < 0);
}

double SLP::step(double x)
{
	return (x >= 0);
}

