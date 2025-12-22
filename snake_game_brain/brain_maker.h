# ifndef BRAIN_MAKER_DOT_H
# define BRAIN_MAKER_DOT_H

typedef struct Neuron {
	float* data;
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

float fire_neuron(Neuron n);
void compute_layer(Layer layer);
float* predict(Model model);
float clamp(float value, float min, float max);
Model create_model(float* fruit_loc_x, float* fruit_loc_y, float* fruit_infront, float* fruit_left, float* fruit_right, float* self_infront, float* self_left, float* self_right, float* nothing_infront, float* nothing_left, float* nothing_right, float* facing_up, float* facing_down, float* facing_left, float* facing_right);
Model mutate_model(Model model);
Model backpropagate_model(Model model, float* expected, float* output, float eta);

#endif // !BRAIN_MAKER_DOT_H
