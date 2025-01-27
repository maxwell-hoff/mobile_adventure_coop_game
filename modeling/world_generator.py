#!/usr/bin/env python3

import random
import math
import yaml

def generate_hex_neighbors(q, r):
    """
    Returns the 6 neighboring hex coordinates in an axial (q, r) system.
    """
    return [
        (q + 1, r),     # East
        (q - 1, r),     # West
        (q, r + 1),     # Southeast
        (q, r - 1),     # Northwest
        (q + 1, r - 1), # Northeast
        (q - 1, r + 1)  # Southwest
    ]

def get_frontier_hexes(used_hexes):
    """
    Returns a set of all hexes that are adjacent to at least one used hex
    but are themselves not used.
    """
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
    Returns the list of hexes if successful, or None if we fail
    to grow to the required size.
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
            # Ensure we don't overlap with used or local
            if (nq, nr) not in used_hexes and (nq, nr) not in local_used:
                # shape_variability ~ probability to skip adding a new neighbor
                if random.random() < (1.0 - shape_variability):
                    continue
                region_hexes.append((nq, nr))
                local_used.add((nq, nr))
                frontier.append((nq, nr))
                added_any = True

                if len(region_hexes) == region_size:
                    break

        # If none were added from 'current', remove it from frontier
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
    max_attempts=1000,
    connect_to_existing=False,
    seed_hex=None
):
    """
    Generates a single region with BFS-based shape generation. 
    
    - If 'seed_hex' is given, we start from that exact coordinate.
    - Else if 'connect_to_existing' is True, we pick a seed from the frontier 
      (ensuring the region touches existing hexes).
    - Otherwise, we pick a random frontier or random hex from a broad range.

    :param region_id: ID for the new region
    :param used_hexes: A set of (q, r) hexes already used by other regions
    :param min_region_hexes, max_region_hexes: Minimum/maximum region size
    :param avg_region_hexes, std_region_hexes: For normal-dist sampling of region size
    :param shape_variability: float (0–1), controlling how “irregular” the shape grows
    :param max_attempts: Max BFS attempts before giving up
    :param connect_to_existing: If True, we must place region so it touches used_hexes
    :param seed_hex: If provided, we forcibly place the region's seed on this hex
    :return: Dict with 'regionId', 'worldHexes', and 'pointsOfInterest'
    """

    # 1) Determine region size from normal distribution
    raw_size = int(round(random.gauss(avg_region_hexes, std_region_hexes)))
    region_size = max(min_region_hexes, min(max_region_hexes, raw_size))

    for attempt in range(max_attempts):
        print(f"Attempt {attempt} of {max_attempts}")
        if seed_hex is not None:
            # Use the forced seed hex (like (0,0) for the first region)
            if seed_hex in used_hexes:

                # If forced seed is somehow used, fail early 
                raise RuntimeError(f"Forced seed {seed_hex} is already in use!")
            chosen_seed = seed_hex
        else:
            if connect_to_existing:
                # We must pick a seed from the frontier
                frontier_hexes = get_frontier_hexes(used_hexes)
                if not frontier_hexes:
                    raise RuntimeError("No available frontier hexes to keep the map connected!")
                chosen_seed = random.choice(list(frontier_hexes))
            else:
                # If there's no frontier, place it randomly in some range
                frontier_hexes = get_frontier_hexes(used_hexes)
                if frontier_hexes:
                    chosen_seed = random.choice(list(frontier_hexes))
                else:
                    chosen_seed = (random.randint(-10000, 10000),
                                   random.randint(-10000, 10000))

            # If chosen_seed is used, skip
            if chosen_seed in used_hexes:
                continue

        # Attempt BFS growth from chosen_seed
        region_candidate = grow_region_bfs(chosen_seed, region_size, used_hexes, shape_variability)
        if region_candidate is not None:
            # Mark used
            for hx in region_candidate:
                used_hexes.add(hx)

            # --- NEW: Add random Points of Interest ---
            # We want 2..5 points, but ensure we don't exceed len(region_candidate)
            poi_count = min(random.randint(2, 5), len(region_candidate))
            # sample distinct hexes from region_candidate
            poi_hexes = random.sample(region_candidate, poi_count)

            points_of_interest = []
            for (q, r) in poi_hexes:
                # Just store minimal info for now (q, r). 
                # (We could also add a random 'type' or so.)
                points_of_interest.append({
                    'q': q,
                    'r': r
                })

            return {
                'regionId': region_id,
                'name': f"Generated Region {region_id}",
                'worldHexes': [{'q': q, 'r': r} for q, r in region_candidate],
                # Store the POIs in region data
                'pointsOfInterest': points_of_interest
            }

    raise RuntimeError(f"Failed to place region {region_id} after {max_attempts} attempts")

def generate_regions_yaml(
    num_regions=100,
    min_region_hexes=2,
    max_region_hexes=8,
    avg_region_hexes=5,
    std_region_hexes=2,
    shape_variability=0.3,
    seed=None
):
    """
    Generates a YAML string containing 'num_regions' regions with:
      - No hex overlaps 
      - No “island” regions (each new region must connect to at least one existing region)
      - One region (the first) explicitly containing (0,0)
      - Region size distribution from normal(avg_region_hexes, std_region_hexes)
      - BFS-based shape generation with shape_variability controlling irregularities
      - 2-5 random Points of Interest in each region
    """
    if seed is not None:
        random.seed(seed)

    used_hexes = set()
    regions = []

    for i in range(num_regions):
        region_id = i + 1
        print(f"Generating region {region_id} of {num_regions}")
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
                connect_to_existing=False,  # Not needed for the first region
                seed_hex=(0, 0)  # Ensure (0,0) is included
            )
        else:
            # For subsequent regions, must be connected to an existing region
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

    # Output as a YAML structure
    return yaml.dump({'regions': regions}, sort_keys=False)

if __name__ == "__main__":
    # Example usage
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
    print("Generated data/gen_world.yaml with random POIs!")
