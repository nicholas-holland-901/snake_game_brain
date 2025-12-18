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
# define STEP_SIZE 1U

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

// training variables
Model models[500];
Model tops[25];
int body_length_reached[500] = { 4 };
int iteration = 0;
int max_iterations = 50;
int model_number = 0;
int max_models = 500;

// Initialize snake body
int snake_body[2][GRID_SIZE_HEIGHT * GRID_SIZE_WIDTH] = {0};
int snake_body_length[1] = { 4 };
int scx, scy, tx, ty;

// Versions of arrays to pass to model
float f_fruit_loc_x[1] = {-1.0f};
float f_fruit_loc_y[1] = {-1.0f};
float f_cell_up[1] = { 0.0f };
float f_cell_down[1] = { 0.0f };
float f_cell_left[1] = { 0.0f };
float f_cell_right[1] = { 0.0f };

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
	for (int i = 0; i < 4; i++) {
		if (ar[i] > max) {
			max = ar[i];
			index = i;
		}
	}
	switch (index) {
		case 0:
		return UP;
		break;
		case 1:
		return DOWN;
		break;
		case 2:
		return LEFT;
		break;
		case 3:
		return RIGHT;
		break;
	default:
		break;
	}
}

float check_cell(int x, int y) {
	int cx = snake_body[0][0] + x;
	int cy = snake_body[1][0] + y;
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
			return -0.5f;
		}
	}
	if (cx == fruit_loc[0] && cy == fruit_loc[1]) {
		return 0.5f;
	}
	else {
		return 0.0f;
	}
}

float* softmax(float* ar) {
	float sum = 0.0f;
	for (int i = 0; i < 4; i++) {
		sum += exp(ar[i]);
	}
	for (int i = 0; i < 4; i++) {
		ar[i] = exp(ar[i]) / sum;
	}
	return ar;
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

void create_first_generation() {
	for (int i = 0; i < max_models; i++) {
		models[i] = create_model(f_fruit_loc_x, f_fruit_loc_y, f_cell_up, f_cell_down, f_cell_left, f_cell_right);
	}
}

// Runs when program starts
SDL_AppResult SDL_AppInit(void **appstate, int argc, char *argv[]) {
	srand(time(NULL));
	SDL_SetAppMetadata("snake_game", "1.0", "");
	// Initialize library
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		return SDL_APP_FAILURE;
	}
	// Creates basic window and default renderer and checks for success
	if (!SDL_CreateWindowAndRenderer("snake_game", WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_BORDERLESS, &window, &renderer)) {
		return SDL_APP_FAILURE;
	}
	// Sets resolution and rendering mode independent of device to renderer
	SDL_SetRenderLogicalPresentation(renderer, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_LOGICAL_PRESENTATION_LETTERBOX);

	init_snake();
	create_first_generation();

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
			default:
				break;
		}
	}
	return SDL_APP_CONTINUE;
}

void create_next_generation() {
	printf("NEXT GEN\n");
	int scores[25] = { 0 };
	// find top 25 performers from current generation
	for (int i = 0; i < max_models; i++) {
		for (int k = 0; k < 25; k++) {
			if (body_length_reached[i] > scores[k]) {
				scores[k] = body_length_reached[i];
				tops[k] = models[i];
				printf("(%d)\n", body_length_reached[i]);
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
		models[i] = create_model(f_fruit_loc_x, f_fruit_loc_y, f_cell_up, f_cell_down, f_cell_left, f_cell_right);
	}

	for (int i = 0; i < max_models; i++) {
		body_length_reached[i] = 4;
	}

	iteration = 0;
	model_number = 0;
	printf("GEN MADE\n");
}

SDL_AppResult SDL_AppIterate(void *appstate) {

	if (iteration > max_iterations) {
		model_number++;
		iteration = 0;
		init_snake();
	}

	if (model_number >= max_models) {
		init_snake();
		create_next_generation();
	}

	// Set background color and clear
	SDL_SetRenderDrawColor(renderer, 25, 25, 25, SDL_ALPHA_OPAQUE);
	SDL_RenderClear(renderer);
	const Uint64 now = SDL_GetTicks();
	// Update game logic every STEP_SIZE milliseconds
	while (now - last_update >= STEP_SIZE) {
		// Update float versions of arrays
		f_fruit_loc_x[0] = ((float)fruit_loc[0] - (float)snake_body[0][0]) / (float)GRID_SIZE_WIDTH;
		f_fruit_loc_y[0] = ((float)fruit_loc[1] - (float)snake_body[1][0]) / (float)GRID_SIZE_HEIGHT;
		f_cell_up[0] = check_cell(0, -1);
		f_cell_down[0] = check_cell(0, 1);
		f_cell_left[0] = check_cell(-1, 0);
		f_cell_right[0] = check_cell(1, 0);

		// Check if snake alive
		for (int i = 1; i < snake_body_length[0]; i++) {
			if (snake_body[0][0] == snake_body[0][i] && snake_body[1][0] == snake_body[1][i]) {
				init_snake();
				model_number++;
				iteration = 0;
				if (model_number >= max_models) {
					create_next_generation();
				}
			}
		}

		// run prediction model
		float* prediction = softmax(predict(models[model_number]));
		last_dir = convert_to_movement(prediction);
		// Show prediction results
		//printf("=========\n");
		//for (int i = 0; i < 4; i++) {
		//	printf("%f\n", prediction[i]);
		//}
		//printf("=========\n");
		
		//int up = prediction[0] > 0.5f;
		//int y = prediction[1] > 0.5f;
		//printf("%f, %f\n", prediction[0], prediction[1]);
		//last_dir = convert_to_movement(x, y);

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
				init_snake();
				model_number++;
				if (model_number >= max_models) {
					create_next_generation();
				}
				iteration = 0;
			}
		}
		check_fruit();
		last_update += STEP_SIZE;
	}

	// Draw snake
	SDL_SetRenderDrawColor(renderer, 150, 255, 0, SDL_ALPHA_OPAQUE);
	rect.x = snake_body[0][0] * GRID_SQUARE_SIZE;
	rect.y = snake_body[1][0] * GRID_SQUARE_SIZE;
	SDL_RenderFillRect(renderer, &rect);
	SDL_SetRenderDrawColor(renderer, 25, 200, 25, SDL_ALPHA_OPAQUE);
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
