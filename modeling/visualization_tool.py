import pygame
import numpy as np
from rl_training import model, scenario

# Hex settings
HEX_RADIUS = 30
GRID_CENTER = (400, 300)  # Center of display

COLOR_PLAYER = (85, 107, 47)
COLOR_ENEMY = (220, 20, 60)
COLOR_HEX = (200, 200, 200)

# Load saved actions log
actions_log = np.load("actions_log.npy", allow_pickle=True)

# Organize actions log by iterations (episodes)
iterations = []
current_iteration = []

for entry in actions_log:
    current_iteration.append(entry)
    if entry["reward"] == 10:  # End of episode (successful checkmate)
        iterations.append(current_iteration)
        current_iteration = []

if current_iteration:
    iterations.append(current_iteration)  # Add last episode if incomplete


def hex_to_pixel(q, r):
    """ Convert axial hex coordinates to pixel coordinates. """
    x = HEX_RADIUS * (3 / 2) * q
    y = HEX_RADIUS * np.sqrt(3) * (r + q / 2)
    return GRID_CENTER[0] + x, GRID_CENTER[1] + y


def draw_hex_grid(screen, pieces, action_step=None):
    """ Draw the hex grid and pieces. """
    screen.fill((255, 255, 255))  # White background

    for piece in pieces:
        q, r = piece["q"], piece["r"]
        x, y = hex_to_pixel(q, r)
        color = COLOR_PLAYER if piece["side"] == "player" else COLOR_ENEMY
        pygame.draw.circle(screen, COLOR_HEX, (x, y), HEX_RADIUS, 1)
        pygame.draw.circle(screen, color, (x, y), HEX_RADIUS // 2)

    if action_step:
        q, r = action_step["player_move"]
        x, y = hex_to_pixel(q, r)
        pygame.draw.circle(screen, (255, 0, 0), (x, y), HEX_RADIUS // 2)  # Highlighted move


def visualize_training(iterations):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Step-by-Step Visualization")
    clock = pygame.time.Clock()

    iteration_index = 0  # Start with the first iteration (episode)
    step_index = 0  # Start with the first step in the iteration
    running = True
    paused = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    step_index = min(step_index + 1, len(iterations[iteration_index]) - 1)
                if event.key == pygame.K_LEFT:
                    step_index = max(0, step_index - 1)
                if event.key == pygame.K_UP:
                    iteration_index = min(iteration_index + 1, len(iterations) - 1)
                    step_index = 0  # Reset step index for new iteration
                if event.key == pygame.K_DOWN:
                    iteration_index = max(0, iteration_index - 1)
                    step_index = 0  # Reset step index for previous iteration
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not paused:
            step_index = (step_index + 1) % len(iterations[iteration_index])
            pygame.time.wait(300)  # Delay for auto-play (300 ms per step)

        current_iteration = iterations[iteration_index]
        current_step = current_iteration[step_index]

        # Draw current iteration and step
        draw_hex_grid(screen, scenario["pieces"], current_step)
        display_iteration_step_info(screen, iteration_index, step_index, len(current_iteration))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def display_iteration_step_info(screen, iteration_idx, step_idx, total_steps):
    """ Display iteration and step information on the screen. """
    font = pygame.font.SysFont(None, 24)
    iteration_text = f"Iteration: {iteration_idx + 1}/{len(iterations)}"
    step_text = f"Step: {step_idx + 1}/{total_steps}"
    iteration_surface = font.render(iteration_text, True, (0, 0, 0))
    step_surface = font.render(step_text, True, (0, 0, 0))
    screen.blit(iteration_surface, (10, 10))
    screen.blit(step_surface, (10, 40))


visualize_training(iterations)
