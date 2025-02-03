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
        
        # 'g' => user wants to jump to an iteration:
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
        
        elif event.key == pygame.K_d:
            # user wants MCTS debug
            print("Press 'd': View MCTS debug info for (iteration, step).")
            try:
                it_input = input("Enter iteration #: ")
                st_input = input("Enter step #: ")
                it_idx = int(it_input) - 1
                st_idx = int(st_input) - 1
                # Now let's try to get that step_data
                if 0 <= it_idx < len(all_iterations):
                    steps = all_iterations[it_idx]
                    if 0 <= st_idx < len(steps):
                        sdata = steps[st_idx]
                        debug = sdata.get("mcts_debug", None)
                        if debug is None:
                            print("No MCTS debug data for that step.")
                        else:
                            print(f"\nMCTS debug for Iter={it_idx+1}, Step={st_idx+1}:")
                            for k,v in debug.items():
                                if k == "chosen_action_idx":
                                    print(f"  chosen_action_idx = {v}")
                                else:
                                    print(f"  action {k} => visits={v['visits']}, q_value={v['q_value']}")
                        input("Press Enter to continue...")
                    else:
                        print("Step out of range.")
                else:
                    print("Iteration out of range.")
            except ValueError:
                print("Invalid integer input. Ignoring...")

def update_from_step_data(step_data):
    """Pull subGridRadius + blockedHexes from step_data into scenario."""
    new_radius = step_data.get("grid_radius", scenario["subGridRadius"])
    new_blocked = step_data.get("blocked_hexes", scenario["blockedHexes"])

    scenario["subGridRadius"] = new_radius
    scenario["blockedHexes"] = new_blocked

def update_piece_positions(step_data):
    """Update piece positions and state from step data.
    
    This function needs to:
    1. Update positions for pieces that are alive
    2. Remove pieces that are dead (q,r = 9999)
    3. Ensure piece labels and classes are preserved
    """
    player_pos = step_data["positions"]["player"]
    enemy_pos = step_data["positions"]["enemy"]
    new_pieces = []
    p_idx = 0
    e_idx = 0

    # First, map current pieces by their original position
    original_pieces = {(p["q"], p["r"]): p for p in scenario["pieces"]}

    # Update player pieces
    for pos in player_pos:
        q, r = float(pos[0]), float(pos[1])
        if q == 9999 or r == 9999:  # Dead piece
            continue
        # Try to find matching piece
        piece = original_pieces.get((q, r))
        if piece:
            new_pieces.append(piece)
        else:
            # If no match, create new piece with default values
            new_pieces.append({
                "side": "player",
                "q": q,
                "r": r,
                "class": "Unknown",
                "label": "?",
                "color": "#556b2f"
            })

    # Update enemy pieces
    for pos in enemy_pos:
        q, r = float(pos[0]), float(pos[1])
        if q == 9999 or r == 9999:  # Dead piece
            continue
        # Try to find matching piece
        piece = original_pieces.get((q, r))
        if piece:
            new_pieces.append(piece)
        else:
            # If no match, create new piece with default values
            new_pieces.append({
                "side": "enemy",
                "q": q,
                "r": r,
                "class": "Unknown",
                "label": "?",
                "color": "#dc143c"
            })

    # Update the scenario
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
        print("No episodes in actions_log.npy. Exiting.")
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