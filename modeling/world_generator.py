import random
import math
import yaml
from collections import deque

# =============== HELPER FUNCTIONS ===============

def generate_hex_neighbors(q, r):
    return [
        (q + 1, r),     # East
        (q - 1, r),     # West
        (q, r + 1),     # Southeast
        (q, r - 1),     # Northwest
        (q + 1, r - 1), # Northeast
        (q - 1, r + 1)  # Southwest
    ]

def get_frontier_hexes(used_hexes):
    frontier = set()
    for (q, r) in used_hexes:
        for (nq, nr) in generate_hex_neighbors(q, r):
            if (nq, nr) not in used_hexes:
                frontier.add((nq, nr))
    return frontier

def grow_region_bfs(seed, region_size, used_hexes, shape_variability):
    """
    Grows a region of 'region_size' hexes via BFS-like expansion 
    starting from 'seed', ensuring no overlap with 'used_hexes'.
    """
    region_hexes = [seed]
    local_used = set([seed])
    frontier = [seed]

    while len(region_hexes) < region_size and frontier:
        current = random.choice(frontier)
        neighbors = generate_hex_neighbors(*current)
        random.shuffle(neighbors)

        added_any = False
        for nq, nr in neighbors:
            if (nq, nr) not in used_hexes and (nq, nr) not in local_used:
                # shape_variability = probability to skip adding
                if random.random() < shape_variability:
                    continue
                region_hexes.append((nq, nr))
                local_used.add((nq, nr))
                frontier.append((nq, nr))
                added_any = True

                if len(region_hexes) == region_size:
                    break

        if not added_any:
            frontier.remove(current)

    if len(region_hexes) == region_size:
        return region_hexes
    else:
        return None

def axial_distance(a, b):
    """Return hex (cube) distance between two axial coords."""
    (q1, r1) = a
    (q2, r2) = b
    # Convert axial -> cube:
    s1 = -q1 - r1
    s2 = -q2 - r2
    return (abs(q1-q2) + abs(r1-r2) + abs(s1-s2)) // 2

# =============== NEW: PUZZLE ASSIGNMENT HELPER ===============
def assign_puzzles_to_region(region_dict, puzzles_list):
    """
    Given a region dictionary (with keys 'worldHexes', 'pointsOfInterest' and 'roads'),
    choose available hexes (not used by POIs or roads) and assign a puzzle from the list.
    The function adds a key 'puzzleScenarios' to the region_dict.
    
    For testing we assign one puzzle per region (if one is available).
    You can later modify this to assign more puzzles or use a different policy.
    """
    available_hexes = set((h['q'], h['r']) for h in region_dict['worldHexes'])
    
    # Remove hexes used by POIs
    if "pointsOfInterest" in region_dict:
        for poi in region_dict["pointsOfInterest"]:
            available_hexes.discard((poi["q"], poi["r"]))
    
    # Remove hexes that appear in any road
    if "roads" in region_dict:
        for road in region_dict["roads"]:
            # road is assumed to be a list of {q, r} dicts in the 'path' key
            for hex_dict in road.get("path", []):
                available_hexes.discard((hex_dict["q"], hex_dict["r"]))
    
    # Convert to list so we can randomly select
    available_hexes = list(available_hexes)
    
    # Initialize the puzzles key
    region_dict["puzzleScenarios"] = []
    
    # For testing, if there is at least one puzzle available and at least one available hex,
    # assign one puzzle to a random available hex.
    if puzzles_list and available_hexes:
        # Pop one puzzle from the puzzles_list so that it is not duplicated.
        puzzle = puzzles_list.pop(0)  # you might randomize the order in puzzles_list if desired
        # Choose a random hex from the available ones
        trigger_hex = random.choice(available_hexes)
        # Set the puzzle's trigger hex
        puzzle["triggerHex"] = {"q": trigger_hex[0], "r": trigger_hex[1]}
        region_dict["puzzleScenarios"].append(puzzle)
    
    return region_dict

# =============== REGION GENERATION ===============

def generate_region(
    region_id,
    used_hexes,
    min_region_hexes,
    max_region_hexes,
    avg_region_hexes,
    std_region_hexes,
    shape_variability,
    max_attempts=3000,
    connect_to_existing=False,
    seed_hex=None
):
    """
    Generates a single region with BFS-based shape generation.
    Returns a dict with 'regionId', 'name', 'worldHexes', 'pointsOfInterest', 'roads',
    and (later) 'puzzleScenarios'.
    """
    raw_size = int(round(random.gauss(avg_region_hexes, std_region_hexes)))
    region_size = max(min_region_hexes, min(max_region_hexes, raw_size))

    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1} of {max_attempts}")
        # Use forced seed if provided
        if seed_hex is not None:
            if seed_hex in used_hexes:
                raise RuntimeError(f"Forced seed {seed_hex} is already in use!")
            chosen_seed = seed_hex
        else:
            if connect_to_existing:
                frontier_hexes = get_frontier_hexes(used_hexes)
                if not frontier_hexes:
                    raise RuntimeError("No available frontier hexes to keep map connected!")
                chosen_seed = random.choice(list(frontier_hexes))
            else:
                frontier_hexes = get_frontier_hexes(used_hexes)
                if frontier_hexes:
                    chosen_seed = random.choice(list(frontier_hexes))
                else:
                    chosen_seed = (
                        random.randint(-10000, 10000),
                        random.randint(-10000, 10000)
                    )

            if chosen_seed in used_hexes:
                continue

        region_candidate = grow_region_bfs(chosen_seed, region_size, used_hexes, shape_variability)
        if region_candidate is not None:
            for hx in region_candidate:
                used_hexes.add(hx)

            # Generate random POIs in region
            poi_count = min(random.randint(2, 5), len(region_candidate))
            poi_hexes = random.sample(region_candidate, poi_count)
            points_of_interest = [ {'q': q, 'r': r} for (q, r) in poi_hexes ]

            # Build roads among the POIs
            roads = []
            if len(points_of_interest) > 1:
                roads = build_roads_for_region({
                    "worldHexes": [ {'q': q, 'r': r} for (q, r) in region_candidate ],
                    "pointsOfInterest": points_of_interest
                })
            
            region_data = {
                'regionId': region_id,
                'name': f"Generated Region {region_id}",
                'worldHexes': [ {'q': q, 'r': r} for (q, r) in region_candidate ],
                'pointsOfInterest': points_of_interest,
                'roads': roads
            }
            
            # --- NEW: Assign puzzles to this region ---
            # Load puzzles from a global list if available.
            # (We assume puzzles_list is defined outside; see generate_regions_yaml below.)
            region_data = assign_puzzles_to_region(region_data, puzzles_list)
            
            return region_data

    raise RuntimeError(f"Failed to place region {region_id} after {max_attempts} attempts")

# =============== ROAD GENERATION ===============

def build_roads_for_region(region_data):
    """
    Build roads connecting the region's POIs.
    Uses a simple MST approach and then finds a path using BFS.
    Returns a list of roads, each a dict with 'path': list of {q, r} dicts.
    """
    pois = region_data['pointsOfInterest']
    hexset = set((h['q'], h['r']) for h in region_data['worldHexes'])
    poi_coords = [(p['q'], p['r']) for p in pois]

    # Build all edges between POIs
    edges = []
    for i in range(len(poi_coords)):
        for j in range(i+1, len(poi_coords)):
            c1 = poi_coords[i]
            c2 = poi_coords[j]
            dist = axial_distance(c1, c2)
            edges.append((c1, c2, dist))
    edges.sort(key=lambda e: e[2])

    # Kruskal-like MST
    parent = {}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(b)] = find(a)
    for c in poi_coords:
        parent[c] = c

    mst_edges = []
    for edge in edges:
        c1, c2, d = edge
        if find(c1) != find(c2):
            union(c1, c2)
            mst_edges.append((c1, c2))
        if len(mst_edges) == len(poi_coords) - 1:
            break

    roads = []
    for start, end in mst_edges:
        path_hexes = find_path_in_region(start, end, hexset)
        if path_hexes:
            path_dicts = [{'q': q, 'r': r} for (q, r) in path_hexes]
            roads.append({'path': path_dicts})
    return roads

def find_path_in_region(start, end, region_hexset):
    """
    BFS (for demonstration) to find a path inside region_hexset from start to end.
    Returns a list of (q, r) if found, else None.
    """
    if start == end:
        return [start]

    from collections import deque
    visited = set([start])
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        cur = path[-1]
        for nbr in generate_hex_neighbors(*cur):
            if nbr not in visited and nbr in region_hexset:
                new_path = path + [nbr]
                if nbr == end:
                    return new_path
                queue.append(new_path)
                visited.add(nbr)
    return None

# =============== MAIN WORLD GEN ===============

# Global puzzles_list; we load the puzzles from generated_puzzles.yaml once.
try:
    with open("data/generated_puzzles.yaml", "r") as f:
        puzzles_list = yaml.safe_load(f)
        # If the file contains a dictionary, adjust accordingly.
        if isinstance(puzzles_list, dict) and "puzzles" in puzzles_list:
            puzzles_list = puzzles_list["puzzles"]
        # Shuffle to randomize the order
        random.shuffle(puzzles_list)
except Exception as e:
    puzzles_list = []
    print("Could not load puzzles:", e)

def generate_regions_yaml(
    num_regions=10,
    min_region_hexes=500,
    max_region_hexes=5000,
    avg_region_hexes=1000,
    std_region_hexes=1000,
    shape_variability=0.3,
    seed=None
):
    if seed is not None:
        random.seed(seed)

    used_hexes = set()
    regions = []

    for i in range(num_regions):
        region_id = i + 1
        print(f"Generating region {region_id}")
        if region_id == 1:
            # Force region #1 to include (0,0)
            region_dict = generate_region(
                region_id=region_id,
                used_hexes=used_hexes,
                min_region_hexes=min_region_hexes,
                max_region_hexes=max_region_hexes,
                avg_region_hexes=avg_region_hexes,
                std_region_hexes=std_region_hexes,
                shape_variability=shape_variability,
                connect_to_existing=False,
                seed_hex=(0, 0)
            )
        else:
            region_dict = generate_region(
                region_id=region_id,
                used_hexes=used_hexes,
                min_region_hexes=min_region_hexes,
                max_region_hexes=max_region_hexes,
                avg_region_hexes=avg_region_hexes,
                std_region_hexes=std_region_hexes,
                shape_variability=shape_variability,
                connect_to_existing=True
            )
        regions.append(region_dict)

    return yaml.dump({'regions': regions}, sort_keys=False)

if __name__ == "__main__":
    yaml_output = generate_regions_yaml(
        num_regions=10,
        min_region_hexes=500,
        max_region_hexes=5000,
        avg_region_hexes=1000,
        std_region_hexes=1000,
        shape_variability=0.3,
        seed=42
    )
    with open('data/gen_world.yaml', 'w') as f:
       f.write(yaml_output)
    print("Generated data/gen_world.yaml with random POIs, roads, and puzzles!")