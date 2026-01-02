#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

# define RAND_WEIGHT() (((rand() % 100) / 500.0f) - 0.2f)
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
Model create_model(float* fruit_loc_x, float* fruit_loc_y, float* fruit_infront, float* fruit_left, float* fruit_right, float* self_infront, float* self_left, float* self_right, float* f_dist_infront, float* f_dist_left, float* f_dist_right);
Model mutate_model(struct Model model);
Model backpropagate_model(Model model, float* expected, float* output, float eta);

/*
Structure Plan
---------
Input layer:
11 Neurons:

1) distance to fruit x
2) distance to fruit y
3) is there fruit infront
4) is there fruit left
5) is there fruit right
6) is self infront
7) is self left
8) is self right
9) distance to fruit from infront
10) distance to fruit from left
11) distance to fruit from right

First Hidden Layer:
32 Neurons
Inputs: 11 values

Second Hidden Layer
32 Neurons
Inputs: 32 values

Output Layer:
3 Neurons:
Inputs: 32 values
Each neuron represents a direction: STRAIGHT, LEFT, RIGHT
Outputs move predicted move quality of each direction, take highest value as action

*/

float fire_neuron(Neuron n) {
	float sum = 0.0f;
	for (int i = 0; i < n.size; i++) {
		sum += n.data[i] * n.weights[i];
	}
	sum += n.bias;
	if (n.activation) {
		// leaky relu activation function
		if (sum < 0) {
			return sum * 0.01f;
		}
		else {
			return sum;
		}
	}
	else {
		// return raw value for softmax activation function on output layer to keep sum of move probabilities equal to 1
		return sum;
	}
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

Model create_model(float* fruit_loc_x, float* fruit_loc_y, float* fruit_infront, float* fruit_left, float* fruit_right, float* self_infront, float* self_left, float* self_right, float* f_dist_infront, float* f_dist_left, float* f_dist_right) {
	Model model;
	model.layers = malloc(4 * sizeof(Layer));
	model.size = 4;
	// Create input layer
	model.layers[0].neurons = malloc(11 * sizeof(Neuron));
	model.layers[0].output = malloc(11 * sizeof(float));
	model.layers[0].size = 11;
	model.layers[0].neurons[0] = create_neuron(fruit_loc_x, 1, false, true);
	model.layers[0].neurons[1] = create_neuron(fruit_loc_y, 1, false, true);
	model.layers[0].neurons[2] = create_neuron(fruit_infront, 1, false, true);
	model.layers[0].neurons[3] = create_neuron(fruit_left, 1, false, true);
	model.layers[0].neurons[4] = create_neuron(fruit_right, 1, false, true);
	model.layers[0].neurons[5] = create_neuron(self_infront, 1, false, true);
	model.layers[0].neurons[6] = create_neuron(self_left, 1, false, true);
	model.layers[0].neurons[7] = create_neuron(self_right, 1, false, true);
	model.layers[0].neurons[8] = create_neuron(f_dist_infront, 1, false, true);
	model.layers[0].neurons[9] = create_neuron(f_dist_left, 1, false, true);
	model.layers[0].neurons[10] = create_neuron(f_dist_right, 1, false, true);

	// Create first hidden layer
	model.layers[1].neurons = malloc(32 * sizeof(Neuron));
	model.layers[1].output = malloc(32 * sizeof(float));
	model.layers[1].size = 32;
	for (int i = 0; i < 32; i++) {
		model.layers[1].neurons[i] = create_neuron(NULL, 11, true, false);
	}

	// Create second hidden layer
	model.layers[2].neurons = malloc(32 * sizeof(Neuron));
	model.layers[2].output = malloc(32 * sizeof(float));
	model.layers[2].size = 32;
	for (int i = 0; i < 32; i++) {
		model.layers[2].neurons[i] = create_neuron(NULL, 32, true, false);
	}

	// Create outupt layer
	model.layers[3].neurons = malloc(3 * sizeof(Neuron));
	model.layers[3].output = malloc(3 * sizeof(float));
	model.layers[3].size = 3;
	model.layers[3].neurons[0] = create_neuron(NULL, 32, false, false);
	model.layers[3].neurons[1] = create_neuron(NULL, 32, false, false);
	model.layers[3].neurons[2] = create_neuron(NULL, 32, false, false);
	return model;
}

// Slightly change weights of model for use in evolutionary training
Model mutate_model(Model model) {
	Model temp = model;
	for (int i = 1; i < model.size; i++) {
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

// Assumes 4 layers
Model backpropagate_model(Model model, float* expected, float* output, float eta) {
	// Allocate memory to store result so that model is updated all at once at the end
	float* output_weights = malloc(model.layers[3].size * model.layers[3].neurons[0].size * sizeof(float));
	float* output_bias = malloc(model.layers[3].size * sizeof(float));
	float* output_total = malloc(model.layers[3].size * sizeof(float));

	float* h2_weights = malloc(model.layers[2].size * model.layers[2].neurons[0].size * sizeof(float));
	float* h2_bias = malloc(model.layers[2].size * sizeof(float));
	float* h2_total = malloc(model.layers[2].size * sizeof(float));

	float* h1_weights = malloc(model.layers[1].size * model.layers[1].neurons[0].size * sizeof(float));
	float* h1_bias = malloc(model.layers[1].size * sizeof(float));

	// Calculate new weights for output layer
	for (int i = 0; i < model.layers[3].size; i++) {
		// detot/dwi = detot/dri * dri/dnati * dnati/dwi
		// detot/dri = ri - ri^
		float detot_dri = output[i] - expected[i];
		// dri/dnati = 1.0f
		float dri_dnati = 1.0f;
		for (int w = 0; w < model.layers[3].neurons[i].size; w++) {
			// dnati/dwi = ri from previous layer
			float dnati_dwi = model.layers[3].neurons[i].data[w];
			// wi+ = wi - eta(detot/dwi)
			// model.layers[2].neurons[i].weights[w] -= (eta * (detot_dri * dri_dnati * dnati_dwi)), -1.0f, 1.0f);
			output_weights[(i * model.layers[3].neurons[i].size) + w] = model.layers[3].neurons[i].weights[w] - (eta * (detot_dri * dri_dnati * dnati_dwi));
		}
		output_total[i] = detot_dri;
		// detot/db = detot/dri * dri/dnati * dnati/db, since it is bias, dnati/db = 1
		// model.layers[2].neurons[i].bias -= (eta * detot_dri * dri_dnati), -1.0f, 1.0f);
		output_bias[i] = model.layers[3].neurons[i].bias - (eta * detot_dri);
	}

	// Calculate new weights for second hidden layer (h2)
	for (int i = 0; i < model.layers[2].size; i++) {
		float tot = 0.0f;
		// detot/dwlhi = detot/drlhi * drlhi/dnatlhi * dnatlhi/dwlhi
		// detot/drlhi = sum of dei/drlhi
		// each dei/drlhi = dei/dnati * dnati/drlhi = detot/dri * dri/dnati * L3.neurons[1-3].weights[i], has already been calculated and stored in float output_total
		// Derivative of relu: 1.0f when value is > 0, otherwise 0.1f
		for (int r = 0; r < model.layers[3].size; r++) {
			tot += output_total[r] * model.layers[3].neurons[r].weights[i];
		}
		float drlhi_dnatlhi = model.layers[2].output[i] > 0 ? 1.0f : 0.01f;
		for (int w = 0; w < model.layers[2].neurons[i].size; w++) {
			// dnatlhi/dwlhi = ri from previous layer (input layer)
			float dnatlhi_dwlhi = model.layers[2].neurons[i].data[w];
			// wi+ = wi - eta(detot/dwlhi)
			// model.layers[2].neurons[i].weights[w] -= (eta * (detot_drlhi * drlhi_dnatlhi * dnatlhi_dwlhi)), -1.0f, 1.0f);
			h2_weights[(i * model.layers[2].neurons[i].size) + w] = model.layers[2].neurons[i].weights[w] - (eta * (tot * drlhi_dnatlhi * dnatlhi_dwlhi));
		}
		h2_total[i] = tot * drlhi_dnatlhi;
		// detot/db = detot/drlhi * drlhi/dnatlhi * dnatlhi/dblhi, again, since it is bias, dnatlhi/dblhi = 1
		// model.layers[2].neurons[i].bias -= (eta * detot_drlhi * drlhi_dnatlhi), -1.0f, 1.0f);
		h2_bias[i] = model.layers[2].neurons[i].bias - (eta * h2_total[i]);
	}

	// Calculate new weights for first hidden layer (h1)
	for (int i = 0; i < model.layers[1].size; i++) {
		// detot/dwlhi = detot/drlhi * drlhi/dnatlhi * dnatlhi/dwlhi
		// detot/drlhi = sum of dei/drlhi
		// each dei/drlhi = dei/dnati * dnati/drlhi = detot/dri * dri/dnati * L2.neurons.weights[i], has already been calculated and stored in float h2_total
		// Derivative of relu: 1.0f when value is > 0, otherwise 0.1f
		float h2_sum = 0.0f;
		for (int t = 0; t < model.layers[2].size; t++) {
			h2_sum += h2_total[t] * model.layers[2].neurons[t].weights[i];
		}
		float drlhi_dnatlhi = model.layers[1].output[i] > 0 ? 1.0f : 0.01f;
		for (int w = 0; w < model.layers[1].neurons[i].size; w++) {
			// dnatlhi/dwlhi = ri from previous layer (input layer)
			float dnatlhi_dwlhi = model.layers[1].neurons[i].data[w];
			// wi+ = wi - eta(detot/dwlhi)
			// model.layers[1].neurons[i].weights[w] -= (eta * (detot_drlhi * drlhi_dnatlhi * dnatlhi_dwlhi)), -1.0f, 1.0f);
			h1_weights[(i * model.layers[1].neurons[i].size) + w] = model.layers[1].neurons[i].weights[w] - (eta * (h2_sum * drlhi_dnatlhi * dnatlhi_dwlhi));
		}
		// detot/db = detot/drlhi * drlhi/dnatlhi * dnatlhi/dblhi, again, since it is bias, dnatlhi/dblhi = 1
		// model.layers[1].neurons[i].bias -= (eta * detot_drlhi * drlhi_dnatlhi), -1.0f, 1.0f);
		h1_bias[i] = model.layers[1].neurons[i].bias - (eta * h2_sum * drlhi_dnatlhi);
	}

	// Update output layer
	for (int i = 0; i < model.layers[3].size; i++) {
		for (int w = 0; w < model.layers[3].neurons[i].size; w++) {
			model.layers[3].neurons[i].weights[w] = output_weights[(i * model.layers[3].neurons[i].size) + w];
		}
		model.layers[3].neurons[i].bias = output_bias[i];
	}

	// Update hidden layer 2
	for (int i = 0; i < model.layers[2].size; i++) {
		for (int w = 0; w < model.layers[2].neurons[i].size; w++) {
			model.layers[2].neurons[i].weights[w] = h2_weights[(i * model.layers[2].neurons[i].size) + w];
		}
		model.layers[2].neurons[i].bias = h2_bias[i];
	}

	// Update hidden layer 1
	for (int i = 0; i < model.layers[1].size; i++) {
		for (int w = 0; w < model.layers[1].neurons[i].size; w++) {
			model.layers[1].neurons[i].weights[w] = h1_weights[(i * model.layers[1].neurons[i].size) + w];
		}
		model.layers[1].neurons[i].bias = h1_bias[i];
	}

	free(output_weights);
	free(output_bias);
	free(output_total);
	free(h2_weights);
	free(h2_bias);
	free(h2_total);
	free(h1_weights);
	free(h1_bias);

	return model;
}