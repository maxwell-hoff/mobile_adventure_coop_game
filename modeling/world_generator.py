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

def generate_region(region_id, used_hexes, min_region_hexes, max_region_hexes, 
                    avg_region_hexes, std_region_hexes, shape_variability, 
                    max_coord_abs=10000, max_attempts=1000):
    """
    Generates a single region with a random shape (in terms of hex adjacency),
    ensuring no overlap with previously used hexes.

    :param region_id: ID of the region to generate.
    :param used_hexes: A set of (q, r) hexes already used by other regions.
    :param min_region_hexes: The minimum number of hexes per region.
    :param max_region_hexes: The maximum number of hexes per region.
    :param avg_region_hexes: The mean number of hexes (for normal dist).
    :param std_region_hexes: Standard deviation for the number of hexes.
    :param shape_variability: A float (0.0 - 1.0) controlling how “random” the shape grows.
    :param max_coord_abs: The absolute range in which we pick random centers: -max_coord_abs to +max_coord_abs.
    :param max_attempts: How many times to attempt to place the region if collisions occur.
    :return: A dict with 'regionId' and 'worldHexes'.
    """

    # 1) Determine how many hexes this region will have
    raw_size = int(round(random.gauss(avg_region_hexes, std_region_hexes)))
    region_size = max(min_region_hexes, min(max_region_hexes, raw_size))

    attempts = 0
    while attempts < max_attempts:
        attempts += 1

        # 2) Pick a random starting hex that's not used
        q0 = random.randint(-max_coord_abs, max_coord_abs)
        r0 = random.randint(-max_coord_abs, max_coord_abs)
        if (q0, r0) in used_hexes:
            continue  # pick another

        # We'll try to grow from this point
        region_hexes = []
        region_hexes.append((q0, r0))
        frontier = [(q0, r0)]
        local_used = set()
        local_used.add((q0, r0))

        # 3) Grow until we reach region_size
        while len(region_hexes) < region_size and frontier:
            # Randomly pick from frontier (to add shape variety)
            current = random.choice(frontier)
            neighbors = generate_hex_neighbors(*current)
            random.shuffle(neighbors)

            added_any = False
            for nq, nr in neighbors:
                if (nq, nr) not in used_hexes and (nq, nr) not in local_used:
                    # shape_variability ~ probability to expand to a new neighbor
                    if random.random() < (1.0 - shape_variability):
                        # skip this neighbor to add shape variability
                        continue
                    # Accept neighbor
                    region_hexes.append((nq, nr))
                    local_used.add((nq, nr))
                    frontier.append((nq, nr))
                    added_any = True
                    if len(region_hexes) == region_size:
                        break
            # If we didn't add a neighbor from current, remove it from frontier
            if not added_any:
                frontier.remove(current)

        # Check if we have the full region_size
        if len(region_hexes) == region_size:
            # Mark these hexes as used globally
            for hx in region_hexes:
                used_hexes.add(hx)

            # Return a region dictionary
            return {
                'regionId': region_id,
                'name': f"Generated Region {region_id}",
                'worldHexes': [{'q': q, 'r': r} for q, r in region_hexes]
            }
        
        # Otherwise, we failed to allocate enough contiguous hexes from that start.
        # We'll just try again with a different starting hex.

    # If we get here, we couldn't place a region in max_attempts tries.
    # You may want to raise an error or handle it differently.
    raise RuntimeError(f"Could not place region {region_id} after {max_attempts} attempts")

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
    Generates a YAML string containing 'num_regions' regions
    with no hex overlaps, controlling for region hex distribution and shape.
    """
    if seed is not None:
        random.seed(seed)

    used_hexes = set()  # Global set of used hexes
    regions = []

    for i in range(num_regions):
        region_dict = generate_region(
            region_id=i+1,
            used_hexes=used_hexes,
            min_region_hexes=min_region_hexes,
            max_region_hexes=max_region_hexes,
            avg_region_hexes=avg_region_hexes,
            std_region_hexes=std_region_hexes,
            shape_variability=shape_variability
        )
        regions.append(region_dict)

    # Combine everything under a top-level key (e.g. "regions")
    yaml_structure = {'regions': regions}
    return yaml.dump(yaml_structure, sort_keys=False)

if __name__ == "__main__":
    # Example usage:
    # Generate 100 regions with 2-8 hexes each, average ~5, std=2,
    # shape variability of 0.3, and seed for reproducibility.
    yaml_output = generate_regions_yaml(
        num_regions=100,
        min_region_hexes=2,
        max_region_hexes=8,
        avg_region_hexes=5,
        std_region_hexes=2,
        shape_variability=0.3,
        seed=42  # set to None for non-deterministic runs
    )
    # print(yaml_output)
    with open('data/gen_world.yaml', 'w') as f:
        f.write(yaml_output)
