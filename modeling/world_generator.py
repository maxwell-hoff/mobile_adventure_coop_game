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
    Returns a set of all hexes that are adjacent to at least one used hex,
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
    
    :param seed: The (q, r) coordinate to start from
    :param region_size: Desired number of hexes in this region
    :param used_hexes: set of (q, r) already used by other regions
    :param shape_variability: Probability to *skip* adding a neighbor
    :return: A list of (q, r) hexes belonging to the region, or None if we fail
    """
    region_hexes = [seed]
    local_used = set([seed])
    frontier = [seed]

    while len(region_hexes) < region_size and frontier:
        # Randomly pick one from frontier
        current = random.choice(frontier)
        neighbors = generate_hex_neighbors(*current)
        random.shuffle(neighbors)

        added_any = False
        for nq, nr in neighbors:
            # Check if not already used by the world or by this region
            if (nq, nr) not in used_hexes and (nq, nr) not in local_used:
                # shape_variability = probability to skip adding
                if random.random() < shape_variability:
                    continue
                
                region_hexes.append((nq, nr))
                local_used.add((nq, nr))
                frontier.append((nq, nr))
                added_any = True

                # If we've reached the needed size, break out
                if len(region_hexes) == region_size:
                    break

        # If no new hex was added from 'current', remove it from the frontier
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
    max_attempts=3000,       # Bump this up for large expansions
    connect_to_existing=False,
    seed_hex=None
):
    """
    Generates a single region with BFS-based shape generation.

    :param region_id: ID for the new region
    :param used_hexes: A set of (q, r) hexes already used by other regions
    :param min_region_hexes, max_region_hexes: Min/Max region size
    :param avg_region_hexes, std_region_hexes: For normal-dist sampling of region size
    :param shape_variability: Probability to skip adding a new neighbor in BFS
    :param max_attempts: Maximum BFS attempts before giving up
    :param connect_to_existing: If True, place region so it touches used_hexes
    :param seed_hex: If provided, forcibly place the region's seed on this hex
    :return: Dict with region data (including 'pointsOfInterest')
    """
    # 1) Determine region size from normal distribution (bounded by min/max)
    raw_size = int(round(random.gauss(avg_region_hexes, std_region_hexes)))
    region_size = max(min_region_hexes, min(max_region_hexes, raw_size))

    for attempt in range(max_attempts):
        print(f"Attempt {attempt} of {max_attempts}")
        # If we have a forced seed hex
        if seed_hex is not None:
            if seed_hex in used_hexes:

                raise RuntimeError(f"Forced seed {seed_hex} is already in use!")
            chosen_seed = seed_hex
        else:
            if connect_to_existing:
                # Must pick a seed from frontier to ensure it's connected
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

            # If chosen_seed is already used, try another attempt
            if chosen_seed in used_hexes:
                continue

        # Attempt BFS-based region growth
        region_candidate = grow_region_bfs(chosen_seed, region_size, used_hexes, shape_variability)
        if region_candidate is not None:
            # Mark these hexes as used
            for hx in region_candidate:
                used_hexes.add(hx)

            # Generate random points of interest in this region
            # We want 2–5 distinct POIs if the region has at least 2 hexes
            poi_count = min(random.randint(2, 5), len(region_candidate))
            poi_hexes = random.sample(region_candidate, poi_count)

            points_of_interest = []
            for (q, r) in poi_hexes:
                # Minimal POI structure; you can add more detail as needed
                points_of_interest.append({'q': q, 'r': r})

            return {
                'regionId': region_id,
                'name': f"Generated Region {region_id}",
                'worldHexes': [{'q': q, 'r': r} for q, r in region_candidate],
                'pointsOfInterest': points_of_interest
            }
        # else BFS failed for this attempt; try again

    # If we exhaust all attempts
    raise RuntimeError(f"Failed to place region {region_id} after {max_attempts} attempts")

def generate_regions_yaml(
    num_regions=10,
    min_region_hexes=500,
    max_region_hexes=5000,
    avg_region_hexes=1000,
    std_region_hexes=1000,
    shape_variability=0.3,
    seed=None
):
    """
    Generates a YAML string containing 'num_regions' regions with:
      - No hex overlaps
      - Each new region connects to an existing region (except the first)
      - Region #1 includes (0,0)
      - Region sizes ~ N(avg_region_hexes, std_region_hexes), bounded by [min_region_hexes, max_region_hexes]
      - BFS-based shape generation with 'shape_variability' controlling how often we skip adding neighbors
      - 2-5 random Points of Interest in each region
    """
    if seed is not None:
        random.seed(seed)

    used_hexes = set()
    regions = []

    for i in range(num_regions):
        print(f"Generating region {i + 1} of {num_regions}")
        region_id = i + 1
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
            # All subsequent regions must connect to already-used hexes
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

    # Dump all regions as YAML
    return yaml.dump({'regions': regions}, sort_keys=False)

if __name__ == "__main__":
    # Example usage with the provided parameters
    yaml_output = generate_regions_yaml(
        num_regions=10,
        min_region_hexes=500,
        max_region_hexes=5000,
        avg_region_hexes=1000,
        std_region_hexes=1000,
        shape_variability=0.3,
        seed=42  # for reproducible results
    )
    with open('data/gen_world.yaml', 'w') as f:
        f.write(yaml_output)
    print("Generated data/gen_world.yaml with random POIs!")
