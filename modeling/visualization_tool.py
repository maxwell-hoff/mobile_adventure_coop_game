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

# Hex grid utilities
def hex_to_pixel(q, r):
    x = HEX_RADIUS * (3 / 2) * q
    y = HEX_RADIUS * np.sqrt(3) * (r + q / 2)
    return GRID_CENTER[0] + x, GRID_CENTER[1] + y

def draw_hex_grid(screen, pieces, action_step=None):
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


def visualize_training(actions_log):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Step-by-Step Visualization")
    clock = pygame.time.Clock()

    step_index = 0
    running = True
    paused = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    step_index = min(step_index + 1, len(actions_log) - 1)
                if event.key == pygame.K_LEFT:
                    step_index = max(0, step_index - 1)
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            step_index = (step_index + 1) % len(actions_log)
            pygame.time.wait(500)  # Delay for auto-play

        # Draw grid and pieces at current step
        draw_hex_grid(screen, scenario["pieces"], actions_log[step_index])
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


visualize_training(actions_log)
