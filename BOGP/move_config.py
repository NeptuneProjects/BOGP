from pathlib import Path
import shutil


root = Path("/Users/williamjenkins/Research/Projects/BOGP/Data/localization/experimental/serial_constrained_localization")
dest = root / "queue"
files = root.glob("*/sequential*/seed*/optim.log")

# for f in files:
    # f.unlink()
    # shutil.move(f, dest / f.name)
