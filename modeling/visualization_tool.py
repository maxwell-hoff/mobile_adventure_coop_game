import pygame
import numpy as np
import yaml
import sys
from stable_baselines3 import PPO

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

# Step and iteration tracking
current_step = 0
current_iteration = 0
all_iterations = []  # This will store steps for each iteration

# Convert axial coordinates to pixel coordinates
def hex_to_pixel(q, r):
    try:
        x = HEX_RADIUS * (3 / 2) * q
        y = HEX_RADIUS * np.sqrt(3) * (r + q / 2)
        return GRID_CENTER[0] + x, GRID_CENTER[1] + y
    except Exception as e:
        print(f"Error in hex_to_pixel conversion: q={q}, r={r}")
        print(f"Exception: {e}")
        raise

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
        try:
            q, r = float(piece["q"]), float(piece["r"])  # Convert to float explicitly
            print(f"Processing piece: {piece['label']}, q={q}, r={r}")  # Debug log
            x, y = hex_to_pixel(q, r)
            x, y = int(x), int(y)  # Convert to integers for pygame
            print(f"Converted coordinates: x={x}, y={y}")  # Debug log
            color = COLOR_PLAYER if piece["side"] == "player" else COLOR_ENEMY
            pygame.draw.circle(screen, color, (x, y), HEX_RADIUS // 2)
            label_font = pygame.font.SysFont("Arial", 16)
            label = label_font.render(piece["label"], True, (255, 255, 255))
            screen.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))
        except Exception as e:
            print(f"Error processing piece: {piece}")
            print(f"Exception: {e}")
            raise

# Draw navigation buttons
def draw_buttons(screen):
    button_font = pygame.font.SysFont("Arial", 20)

    # Previous Step Button
    prev_rect = pygame.Rect(50, 550, 100, 40)
    pygame.draw.rect(screen, (200, 200, 200), prev_rect)
    prev_label = button_font.render("← Prev Step", True, (0, 0, 0))
    screen.blit(prev_label, (prev_rect.x + 10, prev_rect.y + 5))

    # Next Step Button
    next_rect = pygame.Rect(650, 550, 100, 40)
    pygame.draw.rect(screen, (200, 200, 200), next_rect)
    next_label = button_font.render("Next Step →", True, (0, 0, 0))
    screen.blit(next_label, (next_rect.x + 10, next_rect.y + 5))

    # Iteration Display
    iteration_label = button_font.render(f"Iteration: {current_iteration + 1}", True, (0, 0, 0))
    screen.blit(iteration_label, (350, 10))

    return prev_rect, next_rect

# Handle navigation
def handle_navigation(event, prev_rect, next_rect):
    global current_step, current_iteration

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if prev_rect.collidepoint(event.pos):
            current_step = max(0, current_step - 1)
        elif next_rect.collidepoint(event.pos):
            current_step = min(len(all_iterations[current_iteration]) - 1, current_step + 1)

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            current_step = max(0, current_step - 1)
        elif event.key == pygame.K_RIGHT:
            current_step = min(len(all_iterations[current_iteration]) - 1, current_step + 1)
        elif event.key == pygame.K_r:
            current_step = 0  # Reset to the start of the iteration



# Apply the moves from the actions log to update piece positions
def update_piece_positions(step_data):
    # Instead of applying step_data["move"] to scenario,
    # just directly copy all positions from step_data["positions"].
    # That means scenario["pieces"] always matches what the environment saw.

    player_pos = step_data["positions"]["player"]  # e.g. shape (N, 2)
    enemy_pos = step_data["positions"]["enemy"]

    player_index = 0
    enemy_index = 0
    for piece in scenario["pieces"]:
        if piece["side"] == "player":
            piece["q"] = player_pos[player_index][0]
            piece["r"] = player_pos[player_index][1]
            player_index += 1
        else:  # side == "enemy"
            piece["q"] = enemy_pos[enemy_index][0]
            piece["r"] = enemy_pos[enemy_index][1]
            enemy_index += 1


# Draw the state at the current step
def render_scenario():
    global current_step, current_iteration, all_iterations

    # Load actions log from the model output
    try:
        actions_log = np.load("actions_log.npy", allow_pickle=True)
    except FileNotFoundError:
        print("actions_log.npy not found. Please run rl_training.py first.")
        sys.exit(1)
    all_iterations = [actions_log]  # Wrap logs in a list so we can iterate

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Hex Puzzle Scenario Navigation")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        draw_hex_grid(screen, scenario["subGridRadius"])
        
        # Draw the state at the current step
        if all_iterations and len(all_iterations[current_iteration]) > current_step:
            step_data = all_iterations[current_iteration][current_step]
            
            # Update positions according to the step data
            update_piece_positions(step_data)
            
            # Draw the updated pieces on the screen
            draw_pieces(screen, scenario["pieces"])
            
            # Log the current step data
            try:
                print(f"Turn: {step_data.get('turn', 'unknown')}, Move: {step_data.get('move', 'unknown')}, Reward: {step_data.get('reward', 0)}")
            except Exception as e:
                print(f"Error displaying step data: {e}")
                print(f"Step data: {step_data}")

        prev_rect, next_rect = draw_buttons(screen)
        handle_navigation(event, prev_rect, next_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


render_scenario()
