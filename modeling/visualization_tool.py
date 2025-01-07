import pygame
import numpy as np
import yaml
from rl_training import model, scenario

# Hex settings
HEX_RADIUS = 30
GRID_CENTER = (400, 300)
COLOR_HEX = (200, 200, 200)
COLOR_BLOCKED_HEX = (0, 0, 0)  # Blocked hexes shown in black
COLOR_PLAYER = (85, 107, 47)
COLOR_ENEMY = (220, 20, 60)

# Load YAML world data
with open("data/world.yaml", "r") as f:
    world_data = yaml.safe_load(f)

scenario = world_data["regions"][0]["puzzleScenarios"][0]
blocked_hexes = {(h["q"], h["r"]) for h in scenario["blockedHexes"]}

# Convert axial coordinates to pixel coordinates
def hex_to_pixel(q, r):
    x = HEX_RADIUS * (3 / 2) * q
    y = HEX_RADIUS * np.sqrt(3) * (r + q / 2)
    return GRID_CENTER[0] + x, GRID_CENTER[1] + y

# Draw hex grid
def draw_hex_grid(screen, subgrid_radius):
    for q in range(-subgrid_radius, subgrid_radius + 1):
        for r in range(-subgrid_radius, subgrid_radius + 1):
            if abs(q + r) <= subgrid_radius:
                x, y = hex_to_pixel(q, r)
                color = COLOR_BLOCKED_HEX if (q, r) in blocked_hexes else COLOR_HEX
                pygame.draw.polygon(screen, color, hex_corners(x, y), 0)
                pygame.draw.polygon(screen, (0, 0, 0), hex_corners(x, y), 2)

# Get hex corners for polygon drawing
def hex_corners(x, y):
    corners = []
    for i in range(6):
        angle_rad = np.pi / 180 * (60 * i + 30)
        corner_x = x + HEX_RADIUS * np.cos(angle_rad)
        corner_y = y + HEX_RADIUS * np.sin(angle_rad)
        corners.append((corner_x, corner_y))
    return corners

# Draw pieces on the hex map
def draw_pieces(screen, pieces):
    for piece in pieces:
        q, r = piece["q"], piece["r"]
        x, y = hex_to_pixel(q, r)
        color = COLOR_PLAYER if piece["side"] == "player" else COLOR_ENEMY
        pygame.draw.circle(screen, color, (x, y), HEX_RADIUS // 2)
        label_font = pygame.font.SysFont("Arial", 16)
        label = label_font.render(piece["label"], True, (255, 255, 255))
        screen.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))

def render_scenario():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Hex Puzzle Scenario")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        draw_hex_grid(screen, scenario["subGridRadius"])
        draw_pieces(screen, scenario["pieces"])
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

def log_pieces(pieces):
    for piece in pieces:
        pos = f"({piece['q']}, {piece['r']})"
        print(f"{piece['class']} ({piece['label']}) at {pos} - Side: {piece['side']}")

log_pieces(scenario["pieces"])


render_scenario()
