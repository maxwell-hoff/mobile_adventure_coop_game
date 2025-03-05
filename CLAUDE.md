# Mobile Adventure Coop Game Development Guide

## Build Commands
- Run the app: `python app.py`
- Generate puzzles: `python modeling/puzzle_generator.v2.py --randomize-radius --radius-min=2 --radius-max=5`
- Train RL model: `python modeling/rl_training.py --approach=ppo --randomize`
- Run visualization: `python modeling/visualization_tool.py`

## Testing
- No formal test framework. Run individual files directly for testing.

## Code Style
- Imports: Standard imports first, then third-party, then local modules
- Use Python type hints when practical
- Variable naming: snake_case for variables/functions, CamelCase for classes
- Error handling: Use try/except with specific exceptions
- Comments: Document complex algorithms, not obvious functionality
- YAML for configuration (world, pieces, puzzles)
- Use f-strings for string formatting
- Prefer descriptive variable names over abbreviations

## puzzle_generator.v2.py Requirements
1. Generate YAML files that align with the puzzle format defined in world.yaml
2. Include adjustable minimum difficulty parameter
3. Each side should have a random number of pieces between min and max parameters
4. Always include exactly one Priest on each side
5. Victory condition: A side wins by killing the opposing side's Priest
6. Turn-based gameplay where sides alternate, with each piece making one action per turn
7. Actions include moving or casting spells according to rules in pieces.yaml
8. Query the PPO model to determine optimal moves for both sides
9. Allow adjustable ordering of piece actions within each turn
10. Implement "forced mate" detection for puzzle validation
11. Support blocked hexes that are impassable
12. Line of sight checks for attacks and spells that require it
13. Multi-target actions for appropriate classes (e.g., AoE spells)
14. Delayed effects for special abilities (e.g., BloodWarden's 2-turn cast)
15. Generate puzzles where the player has a unique winning strategy
16. Apply hex distance calculations for movement and attack ranges
17. Verify solvability of generated puzzles (should have a solution)
18. Set a maximum number of attempts for finding valid puzzles

## rl_training.py Requirements
1. Support multiple training approaches: PPO, MCTS, tree search
2. Implement a custom gym environment for the hex-based puzzle
3. Support action masking to prevent invalid moves
4. Track rewards for piece kills and game outcomes
5. Allow randomization of puzzle parameters:
   - Grid radius
   - Blocked hex positions
   - Piece positions
   - Piece composition
6. Train models to optimize for priest kills
7. Save trained models for future use in puzzle generation
8. Ensure deterministic prediction for consistent puzzle generation
9. Include visualization of training progress
10. Implement proper state transitions between turns
11. Log episode details for analysis
12. Support both fixed and randomized scenarios
13. Allow adjustment of learning parameters
14. Implement MCTS for comparison with PPO