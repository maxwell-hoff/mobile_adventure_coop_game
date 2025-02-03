import pygame
import numpy as np
import yaml
import sys

BASE_HEX_RADIUS = 30
GRID_CENTER = (400, 300)

COLOR_HEX = (200, 200, 200)
COLOR_BLOCKED_HEX = (0, 0, 0)
COLOR_PLAYER = (85, 107, 47)
COLOR_ENEMY = (220, 20, 60)

with open("data/world.yaml", "r") as f:
    world_data = yaml.safe_load(f)

# We will mutate 'scenario' in-place each time we switch steps
scenario = world_data["regions"][0]["puzzleScenarios"][0]

current_step = 0
current_iteration = 0
all_iterations = []

user_clicked_next_step = False
user_clicked_prev_step = False
user_clicked_next_iter = False
user_clicked_prev_iter = False

def hex_to_pixel(q, r, hex_radius):
    x = hex_radius * 1.5 * q
    import math
    y = hex_radius * math.sqrt(3) * (r + q/2)
    return (int(GRID_CENTER[0] + x), int(GRID_CENTER[1] + y))

def hex_corners(x, y, hex_radius):
    corners = []
    import math
    for i in range(6):
        angle_rad = (60*i + 30) * math.pi / 180
        cx = x + hex_radius * math.cos(angle_rad)
        cy = y + hex_radius * math.sin(angle_rad)
        corners.append((cx, cy))
    return corners

def draw_hex_grid(screen, subgrid_radius, blocked_hexes, hex_radius):
    import math
    for q in range(-subgrid_radius, subgrid_radius + 1):
        for r in range(-subgrid_radius, subgrid_radius + 1):
            if abs(q + r) <= subgrid_radius:
                px, py = hex_to_pixel(q, r, hex_radius)
                color = COLOR_BLOCKED_HEX if (q, r) in blocked_hexes else COLOR_HEX
                corners = hex_corners(px, py, hex_radius)
                pygame.draw.polygon(screen, color, corners, 0)
                pygame.draw.polygon(screen, (0, 0, 0), corners, 2)

def draw_pieces(screen, pieces, hex_radius):
    for piece in pieces:
        q, r = piece["q"], piece["r"]
        px, py = hex_to_pixel(q, r, hex_radius)
        color = COLOR_PLAYER if piece["side"] == "player" else COLOR_ENEMY
        pygame.draw.circle(screen, color, (px, py), hex_radius // 2)

        label_str = piece.get("label", "?")
        label_font = pygame.font.SysFont("Arial", 16)
        txt_surface = label_font.render(label_str, True, (255, 255, 255))
        screen.blit(txt_surface, (px - txt_surface.get_width()//2,
                                  py - txt_surface.get_height()//2))

def draw_buttons(screen):
    button_font = pygame.font.SysFont("Arial", 20)

    prev_iter_rect = pygame.Rect(20, 10, 120, 30)
    pygame.draw.rect(screen, (200,200,200), prev_iter_rect)
    screen.blit(button_font.render("← Prev Iter", True, (0,0,0)), (25,15))

    next_iter_rect = pygame.Rect(660, 10, 120, 30)
    pygame.draw.rect(screen, (200,200,200), next_iter_rect)
    screen.blit(button_font.render("Next Iter →", True, (0,0,0)), (665,15))

    prev_step_rect = pygame.Rect(20, 550, 120, 40)
    pygame.draw.rect(screen, (200,200,200), prev_step_rect)
    screen.blit(button_font.render("← Prev Step", True, (0,0,0)), (25,555))

    next_step_rect = pygame.Rect(660, 550, 120, 40)
    pygame.draw.rect(screen, (200,200,200), next_step_rect)
    screen.blit(button_font.render("Next Step →", True, (0,0,0)), (665,555))

    if len(all_iterations) == 0:
        iteration_text = "Iteration: ?/?"
    else:
        iteration_text = f"Iteration: {current_iteration + 1}/{len(all_iterations)}"
    iteration_label = button_font.render(iteration_text, True, (0,0,0))
    screen.blit(iteration_label, (350, 10))

    return prev_iter_rect, next_iter_rect, prev_step_rect, next_step_rect

def handle_navigation(event, pi, ni, ps, ns):
    global current_step, current_iteration
    global user_clicked_next_step, user_clicked_prev_step
    global user_clicked_next_iter, user_clicked_prev_iter

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if pi.collidepoint(event.pos):
            if len(all_iterations) > 0:
                current_iteration = max(0, current_iteration - 1)
            current_step = 0
            user_clicked_next_iter = False
            user_clicked_prev_iter = True
        elif ni.collidepoint(event.pos):
            if len(all_iterations) > 0:
                current_iteration = min(len(all_iterations)-1, current_iteration + 1)
            current_step = 0
            user_clicked_next_iter = True
            user_clicked_prev_iter = False
        elif ps.collidepoint(event.pos):
            if len(all_iterations) > 0:
                old_step = current_step
                current_step = max(0, current_step - 1)
                user_clicked_prev_step = (current_step != old_step)
                user_clicked_next_step = False
        elif ns.collidepoint(event.pos):
            if len(all_iterations) > 0:
                old_step = current_step
                max_step = len(all_iterations[current_iteration]) - 1
                current_step = min(max_step, current_step + 1)
                user_clicked_next_step = (current_step != old_step)
                user_clicked_prev_step = False

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            if len(all_iterations) > 0:
                old_step = current_step
                current_step = max(0, current_step - 1)
                user_clicked_prev_step = (current_step != old_step)
                user_clicked_next_step = False
        elif event.key == pygame.K_RIGHT:
            if len(all_iterations) > 0:
                old_step = current_step
                max_step = len(all_iterations[current_iteration]) - 1
                current_step = min(max_step, current_step + 1)
                user_clicked_next_step = (current_step != old_step)
                user_clicked_prev_step = False
        elif event.key == pygame.K_r:
            current_step = 0

def update_from_step_data(step_data):
    """Pull subGridRadius + blockedHexes from step_data into scenario."""
    new_radius = step_data.get("grid_radius", scenario["subGridRadius"])
    new_blocked = step_data.get("blocked_hexes", scenario["blockedHexes"])

    scenario["subGridRadius"] = new_radius
    scenario["blockedHexes"] = new_blocked

def update_piece_positions(step_data):
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

    try:
        all_episodes = np.load("actions_log.npy", allow_pickle=True)
    except FileNotFoundError:
        print("actions_log.npy not found. Please run rl_training.py first.")
        sys.exit(1)

    all_iterations = list(all_episodes)
    if not all_iterations:
        print("No episodes in actions_log.npy. Exiting.")
        return

    pygame.init()
    screen = pygame.display.set_mode((800,600))
    pygame.display.set_caption("Hex Puzzle Turn-based Visualization")

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill((255,255,255))

        pi, ni, ps, ns = draw_buttons(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            handle_navigation(event, pi, ni, ps, ns)

        if 0 <= current_iteration < len(all_iterations):
            episode_data = all_iterations[current_iteration]
            current_step = min(current_step, len(episode_data)-1)

            if 0 <= current_step < len(episode_data):
                step_data = episode_data[current_step]

                # 1) Update scenario radius + blocked
                update_from_step_data(step_data)

                # 2) Update piece positions
                update_piece_positions(step_data)

                # 3) Build local blocked set
                local_blocked = {(bh["q"], bh["r"]) for bh in scenario["blockedHexes"]}

                # 4) Draw
                sub_r = scenario["subGridRadius"]  # the actual puzzle radius
                draw_hex_grid(screen, sub_r, local_blocked, BASE_HEX_RADIUS)
                draw_pieces(screen, scenario["pieces"], BASE_HEX_RADIUS)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    render_scenario()