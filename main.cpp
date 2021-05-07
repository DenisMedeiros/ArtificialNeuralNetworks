#include <iostream>
#include <vector>

#include "ann.hpp"

int main(int argc, char **argv)
{

	// Define inputs.
	std::vector<std::vector<double>> inputs = {
			{0, 0}, {0, 1}, {1, 0}, {1, 1}
	};

	// Define outputs.
	std::vector<double> desired = {
			0, 0, 0, 1
	};

	// Create Single-Layer perceptron.
	SLP slp(2, true);

	// Train SLP.
	slp.train(inputs, desired, 0.05, 0.001, 1000);

	// Some tests.
	auto t1 = std::vector<double>{0, 0};
	auto t2 = std::vector<double>{0, 1};
	auto t3 = std::vector<double>{1, 0};
	auto t4 = std::vector<double>{1, 1};

	std::cout << slp.process(t1) << std::endl;
	std::cout << slp.process(t2) << std::endl;
	std::cout << slp.process(t3) << std::endl;
	std::cout << slp.process(t4) << std::endl;


	return 0;
}
