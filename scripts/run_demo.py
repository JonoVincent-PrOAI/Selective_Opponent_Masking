import os
import sys

if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
if os.path.abspath("./") not in sys.path:
    sys.path.append(os.path.abspath("./"))

from game_demo import GameDemo

game = GameDemo(scale = [5,5])
