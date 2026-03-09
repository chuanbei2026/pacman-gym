"""
Pac-Man game entry point.

Usage:
    python3 -m pacman_gym             # Play the game (arrow keys)
    python3 -m pacman_gym train       # Train DQN agent
    python3 -m pacman_gym play        # Watch trained agent play
    python3 -m pacman_gym teach       # RL training visualization
"""

import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""

if cmd == "train":
    sys.argv.pop(1)
    from pacman_gym.train import main as train_main
    train_main()
elif cmd == "play":
    sys.argv = [sys.argv[0], "--play"]
    from pacman_gym.train import main as train_main
    train_main()
elif cmd == "teach":
    from pacman_gym.teach import teach
    teach()
else:
    from pacman_gym.main import main
    main()
