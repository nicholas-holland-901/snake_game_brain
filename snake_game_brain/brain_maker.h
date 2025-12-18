# ifndef BRAIN_MAKER_DOT_H
# define BRAIN_MAKER_DOT_H

typedef struct Neuron {
	float* data;
	float* weights;
	int size;
	float bias;
	bool output_neuron;
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

float fire_neuron(Neuron n);
void compute_layer(Layer layer);
float* predict(Model model);
Model create_model(float* fruit_loc_x, float* fruit_loc_y, float* cell_up, float* cell_down, float* cell_left, float* cell_right);
Model mutate_model(Model model);

#endif // !BRAIN_MAKER_DOT_H
