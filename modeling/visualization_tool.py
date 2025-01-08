import pygame
import numpy as np
import yaml
import sys

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
all_iterations = []  # This will store episodes, each is a list of steps
user_clicked_next_step = False
user_clicked_prev_step = False
user_clicked_next_iter = False
user_clicked_prev_iter = False

def hex_to_pixel(q, r):
    try:
        q = float(q)  # Convert to float first
        r = float(r)
        x = HEX_RADIUS * (3 / 2) * q
        y = HEX_RADIUS * np.sqrt(3) * (r + q / 2)
        # Convert to integers and add grid center
        return (int(GRID_CENTER[0] + x), int(GRID_CENTER[1] + y))
    except Exception as e:
        print(f"Error in hex_to_pixel conversion: q={q}, r={r}")
        print(f"Exception: {e}")
        raise

def draw_hex_grid(screen, subgrid_radius):
    for q in range(-subgrid_radius, subgrid_radius + 1):
        for r in range(-subgrid_radius, subgrid_radius + 1):
            if abs(q + r) <= subgrid_radius:
                x, y = hex_to_pixel(q, r)
                color = COLOR_BLOCKED_HEX if (q, r) in blocked_hexes else COLOR_HEX
                corners = hex_corners(x, y)
                pygame.draw.polygon(screen, color, corners, 0)
                pygame.draw.polygon(screen, (0, 0, 0), corners, 2)

def hex_corners(x, y):
    corners = []
    for i in range(6):
        angle_rad = np.pi / 180 * (60 * i + 30)
        corner_x = x + HEX_RADIUS * np.cos(angle_rad)
        corner_y = y + HEX_RADIUS * np.sin(angle_rad)
        corners.append((corner_x, corner_y))
    return corners

def draw_pieces(screen, pieces):
    for piece in pieces:
        try:
            q, r = piece["q"], piece["r"]
            x, y = hex_to_pixel(q, r)  # Now guaranteed to be integers
            color = COLOR_PLAYER if piece["side"] == "player" else COLOR_ENEMY
            pygame.draw.circle(screen, color, (x, y), HEX_RADIUS // 2)
            label_font = pygame.font.SysFont("Arial", 16)
            label = label_font.render(piece["label"], True, (255, 255, 255))
            screen.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))
        except Exception as e:
            print(f"Error processing piece: {piece}")
            print(f"Exception: {e}")
            raise

def draw_buttons(screen):
    button_font = pygame.font.SysFont("Arial", 20)

    # Prev Iteration
    prev_iter_rect = pygame.Rect(20, 10, 120, 30)
    pygame.draw.rect(screen, (200, 200, 200), prev_iter_rect)
    prev_iter_label = button_font.render("← Prev Iter", True, (0, 0, 0))
    screen.blit(prev_iter_label, (prev_iter_rect.x + 5, prev_iter_rect.y + 5))

    # Next Iteration
    next_iter_rect = pygame.Rect(660, 10, 120, 30)
    pygame.draw.rect(screen, (200, 200, 200), next_iter_rect)
    next_iter_label = button_font.render("Next Iter →", True, (0, 0, 0))
    screen.blit(next_iter_label, (next_iter_rect.x + 5, next_iter_rect.y + 5))

    # Prev Step
    prev_step_rect = pygame.Rect(20, 550, 120, 40)
    pygame.draw.rect(screen, (200, 200, 200), prev_step_rect)
    prev_label = button_font.render("← Prev Step", True, (0, 0, 0))
    screen.blit(prev_label, (prev_step_rect.x + 10, prev_step_rect.y + 5))

    # Next Step
    next_step_rect = pygame.Rect(660, 550, 120, 40)
    pygame.draw.rect(screen, (200, 200, 200), next_step_rect)
    next_label = button_font.render("Next Step →", True, (0, 0, 0))
    screen.blit(next_label, (next_step_rect.x + 10, next_step_rect.y + 5))

    # Iteration Display
    iteration_label = button_font.render(f"Iteration: {current_iteration + 1}/{len(all_iterations)}", True, (0, 0, 0))
    screen.blit(iteration_label, (350, 10))

    return prev_iter_rect, next_iter_rect, prev_step_rect, next_step_rect

def handle_navigation(event,
                      prev_iter_rect,
                      next_iter_rect,
                      prev_step_rect,
                      next_step_rect):
    """
    Sets global flags so we only print
    the move if user explicitly clicked Next Step, etc.
    """
    global current_step, current_iteration
    global user_clicked_next_step, user_clicked_prev_step
    global user_clicked_next_iter, user_clicked_prev_iter

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if prev_iter_rect.collidepoint(event.pos):
            current_iteration = max(0, current_iteration - 1)
            current_step = 0
            user_clicked_next_iter = False
            user_clicked_prev_iter = True
        elif next_iter_rect.collidepoint(event.pos):
            current_iteration = min(len(all_iterations) - 1, current_iteration + 1)
            current_step = 0
            user_clicked_next_iter = True
            user_clicked_prev_iter = False
        elif prev_step_rect.collidepoint(event.pos):
            old_step = current_step
            current_step = max(0, current_step - 1)
            user_clicked_prev_step = (current_step != old_step)
            user_clicked_next_step = False
        elif next_step_rect.collidepoint(event.pos):
            old_step = current_step
            current_step = min(len(all_iterations[current_iteration]) - 1, current_step + 1)
            user_clicked_next_step = (current_step != old_step)
            user_clicked_prev_step = False

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            old_step = current_step
            current_step = max(0, current_step - 1)
            user_clicked_prev_step = (current_step != old_step)
            user_clicked_next_step = False
        elif event.key == pygame.K_RIGHT:
            old_step = current_step
            current_step = min(len(all_iterations[current_iteration]) - 1, current_step + 1)
            user_clicked_next_step = (current_step != old_step)
            user_clicked_prev_step = False
        elif event.key == pygame.K_r:
            current_step = 0

def update_piece_positions(step_data):
    """
    Copy positions from the step_data into scenario["pieces"].
    """
    player_pos = step_data["positions"]["player"]
    enemy_pos = step_data["positions"]["enemy"]

    player_index = 0
    enemy_index = 0
    for piece in scenario["pieces"]:
        if piece["side"] == "player":
            piece["q"] = player_pos[player_index][0]
            piece["r"] = player_pos[player_index][1]
            player_index += 1
        else:  # enemy
            piece["q"] = enemy_pos[enemy_index][0]
            piece["r"] = enemy_pos[enemy_index][1]
            enemy_index += 1

def render_scenario():
    global current_step, current_iteration, all_iterations
    global user_clicked_next_step, user_clicked_prev_step
    global user_clicked_next_iter, user_clicked_prev_iter

    # Load the multiple episodes from file
    try:
        # This shape is (n_episodes,) each is a list of steps
        all_episodes = np.load("actions_log.npy", allow_pickle=True)
    except FileNotFoundError:
        print("actions_log.npy not found. Please run rl_training.py first.")
        sys.exit(1)

    # Convert that array to a Python list for easier handling
    all_iterations = list(all_episodes)  # Now each item is an episode
    if len(all_iterations) == 0:
        print("No episodes in actions_log.npy")
        return

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Hex Puzzle Scenario Navigation")
    clock = pygame.time.Clock()

    running = True
    while running:
        # We reset these flags every frame
        user_clicked_next_step = False
        user_clicked_prev_step = False
        user_clicked_next_iter = False
        user_clicked_prev_iter = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            handle_navigation(event, *button_rects)

        # Clear screen
        screen.fill((255, 255, 255))
        draw_hex_grid(screen, scenario["subGridRadius"])

        # Draw iteration/step info if valid
        if (0 <= current_iteration < len(all_iterations)):
            episode_data = all_iterations[current_iteration]
            if (0 <= current_step < len(episode_data)):
                step_data = episode_data[current_step]
                # Update scenario pieces from step_data
                update_piece_positions(step_data)
                # Draw them
                draw_pieces(screen, scenario["pieces"])

                # Print info only if the user *just* clicked next step
                if user_clicked_next_step:
                    print(f"Step {current_step+1}/{len(episode_data)} | "
                          f"Iteration {current_iteration+1}/{len(all_iterations)} | "
                          f"Turn: {step_data.get('turn','?')} | "
                          f"Move: {step_data.get('move','?')} | "
                          f"Reward: {step_data.get('reward','?')}")

        # Draw buttons
        button_rects = draw_buttons(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# Run it
if __name__ == "__main__":
    render_scenario()
