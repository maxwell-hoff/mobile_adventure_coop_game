# Below is a version of your BFS-based region generator script that also adds roads connecting POIs.
# We'll store roads in each region's dictionary under `roads: [...]`.
# This example approach:
# 1) After we pick random POIs, we connect them with a simple Minimum Spanning Tree approach.
# 2) For each pair of POIs chosen in the MST, we find a path of hexes inside the region.
#    We do a BFS path among regionHexes from one POI to the other.
# 3) We store each path as a list of (q, r) in region['roads']. That way you can see which hexes are used as roads.
#
# This is just one approach. Feel free to adapt or use a more advanced method (e.g., A* with terrain costs, or a graph approach spanning multiple regions).
#

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
    Returns a dict with 'regionId', 'name', 'worldHexes', 'pointsOfInterest', and (later) 'roads'.
    """
    raw_size = int(round(random.gauss(avg_region_hexes, std_region_hexes)))
    region_size = max(min_region_hexes, min(max_region_hexes, raw_size))

    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1} of {max_attempts}")
        # If forced seed
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
                # If no connection required, try frontier if it exists; else random
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

            # We'll build roads in a moment.

            region_data = {
                'regionId': region_id,
                'name': f"Generated Region {region_id}",
                'worldHexes': [ {'q': q, 'r': r} for (q, r) in region_candidate ],
                'pointsOfInterest': points_of_interest,
                'roads': []  # we will fill this after we build roads
            }

            # Now build roads among the POIs
            if len(points_of_interest) > 1:
                roads = build_roads_for_region(region_data)
                region_data['roads'] = roads

            return region_data

    raise RuntimeError(f"Failed to place region {region_id} after {max_attempts} attempts")

# =============== ROAD GENERATION ===============

def build_roads_for_region(region_data):
    """
    Build roads that connect the region's POIs in a logical way.
    We'll do a simple MST approach among POIs to ensure we have at least one path connecting them.
    Then for each pair in the MST, we'll BFS to find a path within the region.

    Returns a list of roads, each road is a dict with 'path': [ {q, r}, ... ].
    """
    pois = region_data['pointsOfInterest']  # e.g. [ {'q':1,'r':2}, ... ]
    hexset = set((h['q'], h['r']) for h in region_data['worldHexes'])

    # Convert POIs to a list of (q, r)
    poi_coords = [(p['q'], p['r']) for p in pois]

    # Build a simple MST among the POIs using e.g. Prim's or Kruskal's.
    # We'll define a basic distance function (axial distance) and connect them.

    # 1) We'll store edges as ((q1, r1), (q2, r2), distance)
    edges = []
    for i in range(len(poi_coords)):
        for j in range(i+1, len(poi_coords)):
            c1 = poi_coords[i]
            c2 = poi_coords[j]
            dist = axial_distance(c1, c2)
            edges.append((c1, c2, dist))
    # 2) Sort edges by distance
    edges.sort(key=lambda e: e[2])

    # 3) We'll do a Kruskal-like MST
    parent = {}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a,b):
        ra = find(a)
        rb = find(b)
        parent[rb] = ra

    for c in poi_coords:
        parent[c] = c

    mst_edges = []
    for e in edges:
        (c1, c2, dist) = e
        if find(c1) != find(c2):
            union(c1,c2)
            mst_edges.append((c1,c2))
        # If we have enough edges: (POIs-1) edges is enough to connect
        if len(mst_edges) == len(poi_coords)-1:
            break

    # Now we have pairs of POIs to connect
    # We'll BFS a path for each pair to get a list of hexes in the road.
    # We'll store each path in 'roads'.
    roads = []
    for (start, end) in mst_edges:
        path_hexes = find_path_in_region(start, end, hexset)
        if path_hexes:
            # convert path to list of dict
            path_dicts = [{'q': q, 'r': r} for (q, r) in path_hexes]
            roads.append({ 'path': path_dicts })

    return roads


def find_path_in_region(start, end, region_hexset):
    """
    BFS or A* to find a path inside region_hexset from start to end.
    We'll do a simple BFS for demonstration.
    Returns list of (q, r) if found, else None.
    """
    if start == end:
        return [start]

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
    return None  # no path found, which is odd if region is contiguous


def axial_distance(a, b):
    """Return hex distance (cube distance) between two axial coords."""
    (q1, r1) = a
    (q2, r2) = b
    # convert axial->cube
    s1 = -q1 - r1
    s2 = -q2 - r2
    return (abs(q1-q2) + abs(r1-r2) + abs(s1-s2)) // 2

# =============== MAIN WORLD GEN ===============

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
    print("Generated data/gen_world.yaml with random POIs and roads!")
