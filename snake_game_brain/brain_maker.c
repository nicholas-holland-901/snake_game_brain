#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

# define RAND_WEIGHT() (((rand() % 1000) / 500.0f) - 1.0f)
# define MUT_CHANCE() ((rand() % 100) < 3)

typedef struct Neuron {
	float *data;
	float* weights;
	int size;
	float bias;
	bool activation;
} Neuron;

typedef struct Layer {
	struct Neuron* neurons;
	float* output;
	int size;
} Layer;

typedef struct Model {
	struct Layer* layers;
	int size;
} Model;

float fire_neuron(struct Neuron);
void compute_layer(struct Layer);
float* predict(struct Model);
Model create_model(float* fruit_loc_x, float* fruit_loc_y, float* cell_up, float* cell_down, float* cell_left, float* cell_right);
Model mutate_model(struct Model model);
Model backpropagate_model(Model model, float* expected, float* output, float eta);

/*
Structure
---------
Input layer:
6 Neurons:
(normalized to be between -1 and 1)
a) distance to fruit_x 
b) distance to fruit_y
stats about 4 cells around head (0 = empty, -1 = body, 1 = fruit)
c) CELL UP
d) CELL DOWN
e) CELL LEFT
f) CELL RIGHT

First Hidden Layer:
6 Neurons
Inputs: 6 values

Second Hidden Layer
6 Neurons
Inputs: 6 values

Output Layer:
4 Neurons:
Inputs: 4 values
Each neuron represents a direction

*/

float fire_neuron(Neuron n) {
	float sum = 0.0f;
	for (int i = 0; i < n.size; i++) {
		sum += n.data[i] * n.weights[i];
	}
	sum += n.bias;
	if (n.activation) {
		// sigmoid activation function if not output neuron
		// return 1.0 / (1.0 + exp(-sum));
		// relu activation function
		return fmaxf(0, sum);
	}
	else {
		// return raw value for softmax activation function on output layer to keep sum of move probabilities equal to 1
		return sum;
	}
}

float clamp(float value, float min, float max) {
	value = fmaxf(value, min);
	value = fminf(value, max);
	return value;
}

void compute_layer(Layer layer) {
	for (int i = 0; i < layer.size; i++) {
		layer.output[i] = fire_neuron(layer.neurons[i]);
	}
}

// Returns two integers (either 0 or 1 each)
float* predict(Model model) {
	for (int i = 0; i < model.size - 1; i++) {
		compute_layer(model.layers[i]);
		for (int j = 0; j < model.layers[i + 1].size; j++) {
			model.layers[i + 1].neurons[j].data = model.layers[i].output;
		}
	}
	compute_layer(model.layers[model.size - 1]);
	return model.layers[model.size - 1].output;
}

Neuron create_neuron(float* data, int size, bool activation, bool input_neuron) {
	Neuron neuron;
	neuron.data = data;
	neuron.weights = malloc(size * sizeof(float));
	neuron.size = size;
	if (input_neuron) {
		neuron.bias = 0.0f;
	}
	else {
		neuron.bias = RAND_WEIGHT();
	}
	neuron.activation = activation;
	for (int i = 0; i < size; i++) {
		if (input_neuron) {
			neuron.weights[i] = 1.0f;
		}
		else {
			neuron.weights[i] = RAND_WEIGHT();
		}
	}
	return neuron;
}

Model create_model(float* fruit_loc_x, float* fruit_loc_y, float* cell_up, float* cell_down, float* cell_left, float* cell_right) {
	Model model;
	model.layers = malloc(3 * sizeof(Layer));
	model.size = 3;
	// Create input layer
	model.layers[0].neurons = malloc(6 * sizeof(Neuron));
	model.layers[0].output = malloc(6 * sizeof(float));
	model.layers[0].size = 6;
	model.layers[0].neurons[0] = create_neuron(fruit_loc_x, 1, false, true);
	model.layers[0].neurons[1] = create_neuron(fruit_loc_y, 1, false, true);
	model.layers[0].neurons[2] = create_neuron(cell_up, 1, false, true);
	model.layers[0].neurons[3] = create_neuron(cell_down, 1, false, true);
	model.layers[0].neurons[4] = create_neuron(cell_left, 1, false, true);
	model.layers[0].neurons[5] = create_neuron(cell_right, 1, false, true);

	// Create first hidden layer
	model.layers[1].neurons = malloc(128 * sizeof(Neuron));
	model.layers[1].output = malloc(128 * sizeof(float));
	model.layers[1].size = 128;
	for (int i = 0; i < 128; i++) {
		model.layers[1].neurons[i] = create_neuron(NULL, 6, true, false);
	}

	// Create second hidden layer
	//model.layers[2].neurons = malloc(6 * sizeof(Neuron));
	//model.layers[2].output = malloc(6 * sizeof(float));
	//model.layers[2].size = 6;
	//for (int i = 0; i < 6; i++) {
	//	model.layers[2].neurons[i] = create_neuron(NULL, 8, false);
	//}

	// Create outupt layer
	model.layers[2].neurons = malloc(4 * sizeof(Neuron));
	model.layers[2].output = malloc(4 * sizeof(float));
	model.layers[2].size = 4;
	model.layers[2].neurons[0] = create_neuron(NULL, 128, false, false);
	model.layers[2].neurons[1] = create_neuron(NULL, 128, false, false);
	model.layers[2].neurons[2] = create_neuron(NULL, 128, false, false);
	model.layers[2].neurons[3] = create_neuron(NULL, 128, false, false);
	return model;
}

// Slightly change weights of model for use in evolutionary training
Model mutate_model(Model model) {
	Model temp = model;
	for (int i = 0; i < model.size; i++) {
		for (int j = 0; j < model.layers[i].size; j++) {
			for (int k = 0; k < model.layers[i].neurons[j].size; k++) {
				if (MUT_CHANCE()) {
					temp.layers[i].neurons[j].weights[k] += ((rand() % 1000) / 1000.0f) - 0.5f;
				}
			}
			if (MUT_CHANCE()) {
				temp.layers[i].neurons[j].bias = ((rand() % 1000) / 1000.0f) - 0.5f;
			}
		}
	}
	return temp;
}

// Assumes 3 layers
Model backpropagate_model(Model model, float* expected, float* output, float eta) {
	Model temp = model;
	// Calculate new weights for output layer
	for (int i = 0; i < temp.layers[2].size; i++) {
		// detot/dwi = detot/dri * dri/dnati * dnati/dwi
		// detot/dri = ri - ri^
		float detot_dri = output[i] - expected[i];
		// dri/dnati = ri(1-ri)
		float dri_dnati = output[i] * (1 - output[i]);
		for (int w = 0; w < temp.layers[2].neurons[i].size; w++) {
			// dnati/dwi = ri from previous layer
			float dnati_dwi = temp.layers[2].neurons[i].data[w];
			// wi+ = wi - eta(detot/dwi)
			temp.layers[2].neurons[i].weights[w] = clamp(temp.layers[2].neurons[i].weights[w] - (eta * (detot_dri * dri_dnati * dnati_dwi)), -1.0f, 1.0f);
		}
		// detot/db = detot/dri * dri/dnati * dnati/db, since it is bias, dnati/db = 1
		temp.layers[2].neurons[i].bias = clamp(temp.layers[2].neurons[i].bias - (eta * detot_dri * dri_dnati), -1.0f, 1.0f);
	}

	// Calculate new weights for hidden layer
	for (int i = 0; i < temp.layers[1].size; i++) {
		// detot/dwlhi = detot/drlhi * drlhi/dnatlhi * dnatlhi/dwlhi
		// detot/drlhi = sum of dei/drlhi
		float detot_drlhi = 0.0f;
		for (int e = 0; e < temp.layers[2].size; e++) {
			// each dei/drlhi = dei/dnati * dnati/drlhi = detot/dri * dri/dnati * L3.neurons[1-4].weights[i]
			detot_drlhi += (output[e] - expected[e]) * (output[i] * (1 - output[i])) * temp.layers[2].neurons[e].weights[i];
		}
		// Derivative of relu: 1 when value is > 0, otherwise 0
		float drlhi_dnatlhi = (float)(temp.layers[1].output[i] > 0);
		for (int w = 0; w < temp.layers[1].neurons[i].size; w++) {
			// dnatlhi/dwlhi = ri from previous layer (input layer)
			float dnatlhi_dwlhi = temp.layers[1].neurons[i].data[w];
			// wi+ = wi - eta(detot/dwlhi)
			temp.layers[1].neurons[i].weights[w] = clamp(temp.layers[1].neurons[i].weights[w] - (eta * (detot_drlhi * drlhi_dnatlhi * dnatlhi_dwlhi)), -1.0f, 1.0f);
		}
		// detot/db = detot/drlhi * drlhi/dnatlhi * dnatlhi/dblhi, again, since it is bias, dnatlhi/dblhi = 1
		temp.layers[1].neurons[i].bias = clamp(temp.layers[1].neurons[i].bias - (eta * detot_drlhi * drlhi_dnatlhi), -1.0f, 1.0f);
	}

	return temp;
}