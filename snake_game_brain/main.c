#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
# include "brain_maker.h"

// Constants
# define GRID_SIZE_WIDTH 15U
# define GRID_SIZE_HEIGHT 15U
# define GRID_SQUARE_SIZE 40U
# define WINDOW_WIDTH (GRID_SIZE_WIDTH * GRID_SQUARE_SIZE)
# define WINDOW_HEIGHT (GRID_SIZE_HEIGHT * GRID_SQUARE_SIZE)
# define STEP_SIZE 150U

static SDL_Window* window = NULL;
static SDL_Renderer* renderer = NULL;

int last_update = 0;
SDL_FRect rect;

enum mov_dir {
	UP,
	DOWN,
	LEFT,
	RIGHT
} MovDir;

enum MovDir last_dir = RIGHT;
enum MovDir last_move_dir = RIGHT;

int fruit_loc[2] = { -1 };
Model my_model;

// Evolutionary training variables
Model models[500];
Model tops[25];
int body_length_reached[500] = { 4 };
int iteration = 0;
int max_iterations = 50;
int model_number = 0;
int max_models = 500;

// Backpropagation training variables
bool feed_snake_inputs = false;

// Reinforcement training variables
bool collided = false;
bool eaten_fruit = false;
float past_distance = 0.0f;
float new_distance = 0.0f;
int chosen_index = 0;
int same_move_count = 0;
int max_same_moves = 15;

// Initialize snake body
int snake_body[2][GRID_SIZE_HEIGHT * GRID_SIZE_WIDTH] = {0};
int snake_body_length[1] = { 4 };
int scx, scy, tx, ty;

float qualities[3] = { 0.0f, 0.0f, 0.0f };
float distances[3] = { 0.0f, 0.0f, 0.0f };

// Versions of arrays to pass to model
float f_fruit_loc_x[1] = {-1.0f};
float f_fruit_loc_y[1] = {-1.0f};
float f_fruit_infront[1] = { 0.0f };
float f_fruit_left[1] = { 0.0f };
float f_fruit_right[1] = { 0.0f };
float f_self_infront[1] = { 0.0f };
float f_self_left[1] = { 0.0f };
float f_self_right[1] = { 0.0f };
float f_dist_infront[1] = { 0.0f };
float f_dist_left[1] = { 0.0f };
float f_dist_right[1] = { 0.0f };

void update_snake_cell(int x, int y, int i) {
	snake_body[0][i] = x;
	snake_body[1][i] = y;
}

void check_fruit() {
	if (snake_body[0][0] == fruit_loc[0] && snake_body[1][0] == fruit_loc[1]) {
		fruit_loc[0] = -1;
		fruit_loc[1] = -1;
		update_snake_cell(snake_body[0][snake_body_length[0] - 1], snake_body[1][snake_body_length[0] - 1], snake_body_length[0]);
		snake_body_length[0]++;
		body_length_reached[model_number]++;
		eaten_fruit = true;
	}
	while (fruit_loc[0] == -1 || fruit_loc[1] == -1) {
		int randx = rand() % GRID_SIZE_WIDTH;
		int randy = rand() % GRID_SIZE_HEIGHT;
		for (int i = 0; i < snake_body_length[0]; i++) {
			if (snake_body[0][i] == randx && snake_body[1][i] == randy) {
				randx = -1;
				randy = -1;
			}
		}
		fruit_loc[0] = randx;
		fruit_loc[1] = randy;
	}
}

enum mov_dir convert_to_movement(float* ar) {
	float max = 0.0f;
	int index = 0;
	for (int i = 0; i < 3; i++) {
		if (ar[i] > max) {
			max = ar[i];
			index = i;
		}
	}
	if (chosen_index == index) {
		same_move_count++;
	}
	else {
		same_move_count = 0;
	}
	chosen_index = index;
	switch (index) {
		case 0:
			// RELATIVE STRAIGHT
			return last_dir;
			break;
		case 1:
			// RELATIVE LEFT
			switch (last_dir) {
			case UP:
				return LEFT;
				break;
			case DOWN:
				return RIGHT;
				break;
			case LEFT:
				return DOWN;
				break;
			case RIGHT:
				return UP;
				break;
			default:
				break;
			}
			break;
		case 2:
			// RELATIVE RIGHT
			switch (last_dir) {
			case UP:
				return RIGHT;
				break;
			case DOWN:
				return LEFT;
				break;
			case LEFT:
				return UP;
				break;
			case RIGHT:
				return DOWN;
				break;
			default:
				break;
			}
			break;
	default:
		break;
	}
}

int wrap_x(int x) {
	if (x >= GRID_SIZE_WIDTH) {
		return 0;
	}
	else if (x < 0) {
		return GRID_SIZE_WIDTH - 1;
	}
	else {
		return x;
	}
}

int wrap_y(int y) {
	if (y >= GRID_SIZE_HEIGHT) {
		return 0;
	}
	else if (y < 0) {
		return GRID_SIZE_HEIGHT - 1;
	}
	else {
		return y;
	}
}

int check_cell(int dir, int offset) {
	int cx = snake_body[0][0];
	int cy = snake_body[1][0];

	switch (dir) {
	case(0):
		// FRONT
		if (last_dir == UP) {
			cy - offset;
		}
		else if (last_dir == DOWN) {
			cy + offset;
		}
		else if (last_dir == LEFT) {
			cx - offset;
		}
		else if (last_dir == RIGHT) {
			cx + offset;
		}
		break;
	case(1):
		// LEFT
		if (last_dir == UP) {
			cx - offset;
		}
		else if (last_dir == DOWN) {
			cx + offset;
		}
		else if (last_dir == LEFT) {
			cy + offset;
		}
		else if (last_dir == RIGHT) {
			cy - offset;
		}
		break;
	case(2):
		// RIGHT
		if (last_dir == UP) {
			cx + offset;
		}
		else if (last_dir == DOWN) {
			cx - offset;
		}
		else if (last_dir == LEFT) {
			cy - offset;
		}
		else if (last_dir == RIGHT) {
			cy + offset;
		}
		break;
	default:
		break;
	}
	if (cx > GRID_SIZE_WIDTH - 1) {
		cx = 0;
	}
	else if (cx < 0) {
		cx = GRID_SIZE_WIDTH - 1;
	}
	if (cy > GRID_SIZE_HEIGHT - 1) {
		cy = 0;
	}
	else if (cy < 0) {
		cy = GRID_SIZE_HEIGHT - 1;
	}
	for (int i = 0; i < GRID_SIZE_HEIGHT * GRID_SIZE_WIDTH; i++) {
		if (cx == snake_body[0][i] && cy == snake_body[1][i]) {
			// Self is in cell
			return -1;
		}
	}
	if (cx == fruit_loc[0] && cy == fruit_loc[1]) {
		// Fruit is in cell
		return 1;
	}
	else {
		// Nothing is in cell
		return 0;
	}
}

float* softmax(float* ar) {
	float max = ar[0];
	float sum = 0.0f;
	for (int i = 0; i < 3; i++) {
		if (ar[i] > max) {
			max = ar[i];
		}
	}
	for (int i = 0; i < 3; i++) {
		sum += exp(ar[i] - max);
	}
	for (int i = 0; i < 3; i++) {
		ar[i] = exp(ar[i] - max) / sum;
	}
	return ar;
}

// Not sqrted distance because it's just being used to compare between moves
float get_distance_to_fruit(int x, int y) {
	return (float)abs((int)(fruit_loc[0] - x)) + abs((int)(fruit_loc[1] - y));
}

void update_snake_inputs() {
	f_fruit_loc_x[0] = (float)(fruit_loc[0] - (float)snake_body[0][0]) / GRID_SIZE_WIDTH;
	f_fruit_loc_y[0] = (float)(fruit_loc[1] - (float)snake_body[1][0]) / GRID_SIZE_HEIGHT;

	f_fruit_infront[0] = 0.0f;
	f_self_infront[0] = 1.0f;
	f_dist_infront[0] = 0.0f;
	f_fruit_left[0] = 0.0f;
	f_self_left[0] = 1.0f;
	f_dist_left[0] = 0.0f;
	f_fruit_right[0] = 0.0f;
	f_self_right[0] = 1.0f;
	f_dist_right[0] = 0.0f;
	if (check_cell(0, 1) == 1) {
		f_fruit_infront[0] = 1.0f;
	}
	if (check_cell(0, 1) == -1) {
		f_self_infront[0] = -1.0f;
	}
	if (check_cell(1, 1) == -1) {
		f_self_left[0] = -1.0f;
	}
	if (check_cell(2, 1) == -1) {
		f_self_right[0] = -1.0f;
	}

	if (check_cell(1, 1) == 1) {
		f_fruit_left[0] = 1.0f;
	}

	if (check_cell(2, 1) == 1) {
		f_fruit_right[0] = 1.0f;
	}
}

void init_snake() {
	snake_body_length[0] = 4;
	for (int i = 0; i < snake_body_length[0]; i++) {
		snake_body[0][i] = 0;
		snake_body[1][i] = 0;
	}
	last_dir = RIGHT;
	last_move_dir = RIGHT;
	scx = GRID_SIZE_WIDTH / 2;
	scy = GRID_SIZE_HEIGHT / 2;
	fruit_loc[0] = -1;
	fruit_loc[1] = -1;
	check_fruit();
	update_snake_cell(scx, scy, 0);
	update_snake_cell(scx - 1, scy, 1);
	update_snake_cell(scx - 2, scy, 2);
	update_snake_cell(scx - 3, scy, 3);
	rect.w = GRID_SQUARE_SIZE;
	rect.h = GRID_SQUARE_SIZE;
}

// Used for evolutionary training
void create_first_generation() {
	for (int i = 0; i < max_models; i++) {
		models[i] = create_model(f_fruit_loc_x, f_fruit_loc_y, f_fruit_infront, f_fruit_left, f_fruit_right, f_self_infront, f_self_left, f_self_right, f_dist_infront, f_dist_left, f_dist_right);
	}
}

void update_move_qualities() {
	float current_distance_to_fruit = get_distance_to_fruit(snake_body[0][0], snake_body[1][0]);
	// Get relative direction cell coordinates: forward, left, right, (x, y)
	int cells[3][2];
	if (last_dir == UP) {
		cells[0][0] = 0;
		cells[0][1] = -1;
		cells[1][0] = -1;
		cells[1][1] = 0;
		cells[2][0] = 1;
		cells[2][1] = 0;
	}
	else if (last_dir == DOWN) {
		cells[0][0] = 0;
		cells[0][1] = 1;
		cells[1][0] = 1;
		cells[1][1] = 0;
		cells[2][0] = -1;
		cells[2][1] = 0;
	}
	else if (last_dir == LEFT) {
		cells[0][0] = -1;
		cells[0][1] = 0;
		cells[1][0] = 0;
		cells[1][1] = 1;
		cells[2][0] = 0;
		cells[2][1] = -1;
	}
	else if (last_dir == RIGHT) {
		cells[0][0] = 1;
		cells[0][1] = 0;
		cells[1][0] = 0;
		cells[1][1] = -1;
		cells[2][0] = 0;
		cells[2][1] = 1;
	}
	for (int i = 0; i < 3; i++) {
		cells[i][0] = wrap_x(cells[i][0] + snake_body[0][0]);
		cells[i][1] = wrap_y(cells[i][1] + snake_body[1][0]);
	}
	for (int i = 0; i < 3; i++) {
		// See if new location would be closer to fruit
		if (get_distance_to_fruit(cells[i][0], cells[i][1]) > current_distance_to_fruit) {
			qualities[i] = 0.2f;
			distances[i] = -0.8f;
		}
		else if (get_distance_to_fruit(cells[i][0], cells[i][1]) < current_distance_to_fruit) {
			qualities[i] = 2.0f;
			distances[i] = 1.0f;
		}
		else if (get_distance_to_fruit(cells[i][0], cells[i][1]) == current_distance_to_fruit) {
			qualities[i] = 0.5f;
			distances[i] = 0.1f;
		}
		// Check if locations have fruit
		if (cells[i][0] == fruit_loc[0] && cells[i][1] == fruit_loc[1]) {
			qualities[i] = 2.5f;
		}
		// Check if collision
		int num_x = 0;
		int num_y = 0;
		for (int n = 1; n < snake_body_length[0]; n++) {
			if (cells[i][0] == snake_body[0][n] && cells[i][1] == snake_body[1][n]) {
				qualities[i] = -2.5f; // -0.2f;
				distances[i] = -2.5f;
			}
			// Check if square is surrounded by two snake pieces on either side (high chance of getting stuck in self loop)
			if (cells[i][0] + 1 == snake_body[0][n] && cells[i][1] == snake_body[1][n]) {
				num_x++;
			}
			if (cells[i][0] - 1 == snake_body[0][n] && cells[i][1] == snake_body[1][n]) {
				num_x++;
			}
			if (cells[i][0] == snake_body[0][n] && cells[i][1] + 1 == snake_body[1][n]) {
				num_y++;
			}
			if (cells[i][0] == snake_body[0][n] && cells[i][1] - 1 == snake_body[1][n]) {
				num_y++;
			}
		}
		if (num_x == 2 || num_y == 2) {
			// High chance of getting stuck if follow path
			qualities[i] = -2.5f;
			distances[i] = -2.5f;
		}
	}
}

void disp_prediction(float* prediction) {
	printf("=========\n");
	for (int i = 0; i < 3; i++) {
		printf("%f\n", prediction[i]);
	}
	printf("=========\n");
}

// Runs when program starts
SDL_AppResult SDL_AppInit(void **appstate, int argc, char *argv[]) {
	srand(time(NULL));
	SDL_SetAppMetadata("snake_game", "1.0", "");
	// initialize library
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		return SDL_APP_FAILURE;
	}
	// Creates basic window and default renderer and checks for success
	if (!SDL_CreateWindowAndRenderer("snake_game", WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_BORDERLESS, &window, &renderer)) {
		return SDL_APP_FAILURE;
	}
	// Sets resolution and rendering mode independent of device to renderer
	SDL_SetRenderLogicalPresentation(renderer, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_LOGICAL_PRESENTATION_LETTERBOX);

	my_model = create_model(f_fruit_loc_x, f_fruit_loc_y, f_fruit_infront, f_fruit_left, f_fruit_right, f_self_infront, f_self_left, f_self_right, f_dist_infront, f_dist_left, f_dist_right);

	init_snake();
	// Evolutionary training
	//create_first_generation();

	return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
	if (event->type == SDL_EVENT_QUIT) {
		return SDL_APP_SUCCESS;
	} else if(event->type == SDL_EVENT_KEY_DOWN) {
		switch (event->key.scancode) {
			case SDL_SCANCODE_Q:
				return SDL_APP_SUCCESS;
				break;
			case SDL_SCANCODE_UP:
				last_dir = UP;
				break;
			case SDL_SCANCODE_DOWN:
				last_dir = DOWN;
				break;
			case SDL_SCANCODE_LEFT:
				last_dir = LEFT;
				break;
			case SDL_SCANCODE_RIGHT:
				last_dir = RIGHT;
				break;
			case SDL_SCANCODE_F:
				feed_snake_inputs = !feed_snake_inputs;
			default:
				break;
		}
	}
	return SDL_APP_CONTINUE;
}

// Used for evolutionary training
void create_next_generation() {
	int scores[25] = { 0 };
	// find top 25 performers from current generation
	for (int i = 0; i < max_models; i++) {
		for (int k = 0; k < 25; k++) {
			if (body_length_reached[i] > scores[k]) {
				scores[k] = body_length_reached[i];
				tops[k] = models[i];
				// Print snake length reached
				printf("%d\n", body_length_reached[i]);
				break;
			}
		}
	}
	// keep top 25 models in next generation
	for (int i = 0; i < 25; i++) {
		models[i] = tops[i];
	}
	// create most of next generation based on top performers
	int index = 25;
	for (int i = 1; i < 17; i++) {
		for (int k = 0; k < 25; k++) {
			index++;
			models[index] = mutate_model(tops[k]);
		}
	}
	// create random for some of new generation
	for (int i = 427; i < max_models; i++) {
		models[i] = create_model(f_fruit_loc_x, f_fruit_loc_y, f_fruit_infront, f_fruit_left, f_fruit_right, f_self_infront, f_self_left, f_self_right, f_dist_infront, f_dist_left, f_dist_right);
	}
	for (int i = 0; i < max_models; i++) {
		body_length_reached[i] = 4;
	}
	iteration = 0;
	model_number = 0;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
	// Evolutionary training
	//if (iteration > max_iterations) {
	//	model_number++;
	//	iteration = 0;
	//	init_snake();
	//}
	//if (model_number >= max_models) {
	//	init_snake();
	//	create_next_generation();
	//}

	// Set background color and clear
	SDL_SetRenderDrawColor(renderer, 25, 25, 25, SDL_ALPHA_OPAQUE);
	SDL_RenderClear(renderer);
	const Uint64 now = SDL_GetTicks();
	// Update game logic every STEP_SIZE milliseconds
	int dif = STEP_SIZE;
	if (feed_snake_inputs) {
		dif = 1;
	}
	while (now - last_update >= dif) {
		// Update float versions of arrays
		update_snake_inputs();
		// Calculate quality of each possible move
		update_move_qualities();
		f_dist_infront[0] = distances[0];
		f_dist_left[0] = distances[1];
		f_dist_right[0] = distances[2];
		collided = false;
		eaten_fruit = false;
		past_distance = get_distance_to_fruit(snake_body[0][0], snake_body[1][0]);
		// Check if snake alive
		for (int i = 1; i < snake_body_length[0]; i++) {
			if (snake_body[0][0] == snake_body[0][i] && snake_body[1][0] == snake_body[1][i]) {
				init_snake();
				// Evolutionary training
				//model_number++;
				//iteration = 0;
				//if (model_number >= max_models) {
				//	create_next_generation();
				//}
			}
		}

		// Run model
		float* prediction = softmax(predict(my_model));
		float* expected = softmax(qualities);

		// Backpropagate model based on quality of last move
		if (feed_snake_inputs) {
			my_model = backpropagate_model(my_model, expected, prediction, 0.1f);
			prediction = softmax(predict(my_model));
		}

		// Evolutionary training
		// float* prediction = softmax(predict(models[model_number]));
		
		// Predict next move
		last_dir = convert_to_movement(prediction);
		// Show prediction results
		//disp_prediction(prediction);

		// Move each snake body part forward
		for (int i = snake_body_length[0] - 1; i >  0; i--) {
			update_snake_cell(snake_body[0][i - 1], snake_body[1][i - 1], i);
		}
		// Move snake head in last pressed direction
		tx = snake_body[0][0];
		ty = snake_body[1][0];
		iteration++;
		switch (last_dir) {
			case UP:
				if (last_move_dir == DOWN) {
					ty++;
					last_move_dir = DOWN;
				}
				else {
					ty--;
					last_move_dir = UP;
				}
				break;
			case DOWN:
				if (last_move_dir == UP) {
					ty--;
					last_move_dir = UP;
				}
				else {
					ty++;
					last_move_dir = DOWN;
				}
				break;
			case LEFT:
				if (last_move_dir == RIGHT) {
					tx++;
					last_move_dir = RIGHT;
				}
				else {
					tx--;
					last_move_dir = LEFT;
				}
				break;
			case RIGHT:
				if (last_move_dir == LEFT) {
					tx--;
					last_move_dir = LEFT;
				}
				else {
					tx++;
					last_move_dir = RIGHT;
				}
				break;
			default:
				break;
		}
		// Wrap around x
		if (tx < 0) {tx = GRID_SIZE_WIDTH - 1;}
		else if (tx > GRID_SIZE_WIDTH - 1) {tx = 0;}
		// Wrap around y
		if (ty < 0) {ty = GRID_SIZE_HEIGHT - 1;} 
		else if (ty > GRID_SIZE_HEIGHT - 1) {ty = 0;}
		update_snake_cell(tx, ty, 0);
		// Check for collision after move
		for (int i = 1; i < snake_body_length[0]; i++) {
			if (snake_body[0][i] == snake_body[0][0] && snake_body[1][i] == snake_body[1][0]) {
				// Print snake length reached
				printf("%d\n", snake_body_length[0]);
				init_snake();
				collided = true;
				// Evolutionary training
				//model_number++;
				//if (model_number >= max_models) {
				//	create_next_generation();
				//}
				//iteration = 0;
			}
		}
		check_fruit();
		new_distance = get_distance_to_fruit(snake_body[0][0], snake_body[1][0]);
		last_update += dif;
	}

	// Draw snake
	if (feed_snake_inputs) {
		SDL_SetRenderDrawColor(renderer, 105, 0, 255, SDL_ALPHA_OPAQUE);
	}
	else {
		// Default green snake
		SDL_SetRenderDrawColor(renderer, 150, 255, 0, SDL_ALPHA_OPAQUE);
	}
	rect.x = snake_body[0][0] * GRID_SQUARE_SIZE;
	rect.y = snake_body[1][0] * GRID_SQUARE_SIZE;
	SDL_RenderFillRect(renderer, &rect);
	if (feed_snake_inputs) {
		SDL_SetRenderDrawColor(renderer, 230, 55, 230, SDL_ALPHA_OPAQUE);
	}
	else {
		// Default green snake
		SDL_SetRenderDrawColor(renderer, 25, 200, 25, SDL_ALPHA_OPAQUE);
	}

	for (int i = 1; i < snake_body_length[0]; i++) {
		rect.x = snake_body[0][i] * GRID_SQUARE_SIZE;
		rect.y = snake_body[1][i] * GRID_SQUARE_SIZE;
		SDL_RenderFillRect(renderer, &rect);
	}

	// Draw fruit
	if (fruit_loc[0] != -1 && fruit_loc[1] != -1) {
		SDL_SetRenderDrawColor(renderer, 255, 50, 50, SDL_ALPHA_OPAQUE);
		rect.x = fruit_loc[0] * GRID_SQUARE_SIZE;
		rect.y = fruit_loc[1] * GRID_SQUARE_SIZE;
		SDL_RenderFillRect(renderer, &rect);
	}

	SDL_RenderPresent(renderer);

	return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {

}
