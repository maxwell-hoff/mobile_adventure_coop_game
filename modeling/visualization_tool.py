import pygame
import numpy as np
import yaml
import sys

# Hex settings
HEX_RADIUS = 30
GRID_CENTER = (400, 300)
COLOR_HEX = (200, 200, 200)
COLOR_BLOCKED_HEX = (0, 0, 0)
COLOR_PLAYER = (85, 107, 47)
COLOR_ENEMY = (220, 20, 60)

with open("data/world.yaml", "r") as f:
    world_data = yaml.safe_load(f)

scenario = world_data["regions"][0]["puzzleScenarios"][0]
blocked_hexes = {(h["q"], h["r"]) for h in scenario["blockedHexes"]}

current_step = 0
current_iteration = 0
all_iterations = []

user_clicked_next_step = False
user_clicked_prev_step = False
user_clicked_next_iter = False
user_clicked_prev_iter = False

def hex_to_pixel(q, r):
    x = HEX_RADIUS * (3/2) * float(q)
    y = HEX_RADIUS * (3**0.5) * (float(r) + float(q)/2)
    return (int(GRID_CENTER[0] + x), int(GRID_CENTER[1] + y))

def draw_hex_grid(screen, subgrid_radius):
    for q in range(-subgrid_radius, subgrid_radius+1):
        for r in range(-subgrid_radius, subgrid_radius+1):
            if abs(q + r) <= subgrid_radius:
                x, y = hex_to_pixel(q, r)
                color = COLOR_BLOCKED_HEX if (q, r) in blocked_hexes else COLOR_HEX
                corners = hex_corners(x, y)
                pygame.draw.polygon(screen, color, corners, 0)
                pygame.draw.polygon(screen, (0, 0, 0), corners, 2)

def hex_corners(x, y):
    corners = []
    for i in range(6):
        angle_rad = (60*i + 30) * (np.pi / 180)
        cx = x + HEX_RADIUS*np.cos(angle_rad)
        cy = y + HEX_RADIUS*np.sin(angle_rad)
        corners.append((cx, cy))
    return corners

def draw_pieces(screen, pieces):
    for piece in pieces:
        q, r = piece["q"], piece["r"]
        x, y = hex_to_pixel(q, r)
        color = COLOR_PLAYER if piece["side"] == "player" else COLOR_ENEMY
        pygame.draw.circle(screen, color, (x, y), HEX_RADIUS//2)

        label_font = pygame.font.SysFont("Arial", 16)
        label = piece.get("label","?")
        txt_surface = label_font.render(label, True, (255, 255, 255))
        screen.blit(txt_surface, (x - txt_surface.get_width()//2, y - txt_surface.get_height()//2))

def draw_buttons(screen):
    button_font = pygame.font.SysFont("Arial", 20)

    # Prev Iteration
    prev_iter_rect = pygame.Rect(20, 10, 120, 30)
    pygame.draw.rect(screen, (200,200,200), prev_iter_rect)
    screen.blit(button_font.render("← Prev Iter", True, (0,0,0)), (25,15))

    # Next Iteration
    next_iter_rect = pygame.Rect(660, 10, 120, 30)
    pygame.draw.rect(screen, (200,200,200), next_iter_rect)
    screen.blit(button_font.render("Next Iter →", True, (0,0,0)), (665,15))

    # Prev Step
    prev_step_rect = pygame.Rect(20, 550, 120, 40)
    pygame.draw.rect(screen, (200,200,200), prev_step_rect)
    screen.blit(button_font.render("← Prev Step", True, (0,0,0)), (25,555))

    # Next Step
    next_step_rect = pygame.Rect(660, 550, 120, 40)
    pygame.draw.rect(screen, (200,200,200), next_step_rect)
    screen.blit(button_font.render("Next Step →", True, (0,0,0)), (665,555))

    iteration_label = button_font.render(f"Iteration: {current_iteration + 1}/{len(all_iterations)}", True, (0,0,0))
    screen.blit(iteration_label, (350, 10))

    return prev_iter_rect, next_iter_rect, prev_step_rect, next_step_rect

def handle_navigation(event, pi, ni, ps, ns):
    global current_step, current_iteration
    global user_clicked_next_step, user_clicked_prev_step
    global user_clicked_next_iter, user_clicked_prev_iter

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if pi.collidepoint(event.pos):
            current_iteration = max(0, current_iteration - 1)
            current_step = 0
            user_clicked_next_iter = False
            user_clicked_prev_iter = True
        elif ni.collidepoint(event.pos):
            current_iteration = min(len(all_iterations)-1, current_iteration + 1)
            current_step = 0
            user_clicked_next_iter = True
            user_clicked_prev_iter = False
        elif ps.collidepoint(event.pos):
            old_step = current_step
            current_step = max(0, current_step - 1)
            user_clicked_prev_step = (current_step != old_step)
            user_clicked_next_step = False
        elif ns.collidepoint(event.pos):
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

        elif event.key == pygame.K_g:
            print("Type an iteration number in the console and press Enter.")
            try:
                user_input = input("Enter iteration # to jump to (1-based): ")
                desired_iter = int(user_input) - 1
                if 0 <= desired_iter < len(all_iterations):
                    current_iteration = desired_iter
                    current_step = 0
                else:
                    print(f"Invalid iteration: must be between 1 and {len(all_iterations)}.")
            except ValueError:
                print("Invalid integer input. Ignoring...")

def update_piece_positions(step_data):
    """
    step_data["positions"] has shape:
       {"player": np.array([...]),
        "enemy":  np.array([...])}
    We overwrite scenario["pieces"] to match those positions exactly.
    """
    player_pos = step_data["positions"]["player"]
    enemy_pos = step_data["positions"]["enemy"]

    new_pieces = []
    p_idx = 0
    e_idx = 0

    for piece in scenario["pieces"]:
        if piece["side"] == "player":
            if p_idx < len(player_pos):
                piece["q"] = float(player_pos[p_idx][0])
                piece["r"] = float(player_pos[p_idx][1])
                new_pieces.append(piece)
                p_idx += 1
        else:
            if e_idx < len(enemy_pos):
                piece["q"] = float(enemy_pos[e_idx][0])
                piece["r"] = float(enemy_pos[e_idx][1])
                new_pieces.append(piece)
                e_idx += 1

    scenario["pieces"].clear()
    scenario["pieces"].extend(new_pieces)

def render_scenario():
    global current_iteration, current_step, all_iterations
    global user_clicked_next_step, user_clicked_prev_step
    global user_clicked_next_iter, user_clicked_prev_iter

    try:
        all_episodes = np.load("actions_log.npy", allow_pickle=True)
    except FileNotFoundError:
        print("actions_log.npy not found. Please run rl_training.py first.")
        sys.exit(1)

    all_iterations = list(all_episodes)
    if not all_iterations:
        print("No episodes in actions_log.npy")
        return

    pygame.init()
    screen = pygame.display.set_mode((800,600))
    pygame.display.set_caption("Hex Puzzle Turn-based Visualization")
    clock = pygame.time.Clock()
    running = True

    while running:
        user_clicked_next_step = False
        user_clicked_prev_step = False
        user_clicked_next_iter = False
        user_clicked_prev_iter = False

        screen.fill((255,255,255))
        draw_hex_grid(screen, scenario["subGridRadius"])
        pi, ni, ps, ns = draw_buttons(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            handle_navigation(event, pi, ni, ps, ns)

        if 0 <= current_iteration < len(all_iterations):
            episode_data = all_iterations[current_iteration]
            # clamp current_step in case we clicked next iteration with fewer steps
            current_step = min(current_step, len(episode_data)-1)

            if 0 <= current_step < len(episode_data):
                step_data = episode_data[current_step]
                update_piece_positions(step_data)
                draw_pieces(screen, scenario["pieces"])

                if user_clicked_next_step:
                    print(f"Step {current_step+1}/{len(episode_data)} "
                          f"| Iteration {current_iteration+1}/{len(all_iterations)} "
                          f"| Turn#: {step_data.get('turn_number','?')} "
                          f"| TurnSide: {step_data.get('turn_side','?')} "
                          f"| Reward: {step_data.get('reward','?')}")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    render_scenario()
