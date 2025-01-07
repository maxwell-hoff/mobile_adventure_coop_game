import pygame
import matplotlib.pyplot as plt
import numpy as np
from rl_training import model

# Hex settings
HEX_RADIUS = 30
GRID_CENTER = (400, 300)  # Center of display

# Color settings
COLOR_PLAYER = (85, 107, 47)  # Dark green
COLOR_ENEMY = (220, 20, 60)  # Crimson
COLOR_HEX = (200, 200, 200)


def hex_to_pixel(q, r):
    """ Convert axial hex coordinates to pixel coordinates. """
    x = HEX_RADIUS * (3/2) * q
    y = HEX_RADIUS * np.sqrt(3) * (r + q / 2)
    return GRID_CENTER[0] + x, GRID_CENTER[1] + y


def draw_hex_grid(screen, pieces):
    """ Draw the hex grid and pieces. """
    for piece in pieces:
        q, r = piece["q"], piece["r"]
        x, y = hex_to_pixel(q, r)

        # Draw hex outline
        pygame.draw.circle(screen, COLOR_HEX, (x, y), HEX_RADIUS, 1)

        # Draw player/enemy pieces
        color = COLOR_PLAYER if piece["side"] == "player" else COLOR_ENEMY
        pygame.draw.circle(screen, color, (x, y), HEX_RADIUS // 2)


def visualize_training(model):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Training Visualization")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # White background
        draw_hex_grid(screen, scenario["pieces"])

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def plot_training_logs(rewards, losses):
    """ Plot training rewards and losses. """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Reward')
    plt.title("Reward Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', color='red')
    plt.title("Loss Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


# Example log data for visualization
rewards_log = np.random.randn(200).cumsum()
losses_log = np.abs(np.random.randn(200))

plot_training_logs(rewards_log, losses_log)
visualize_training(model)