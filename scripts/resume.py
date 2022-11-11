# Create logic for resuming simulations:

from pathlib import Path

# BOGP/Data/Simulations/Protected/localization_500iter/acq_func=ExpectedImprovement__snr=20__rec_r=0.5__src_z=62/Runs/0002406475

run = (
    Path().cwd().parent
    / "Data"
    / "Simulations"
    / "Protected"
    / "localization_500iter"
    / "acq_func=ExpectedImprovement__snr=20__rec_r=0.5__src_z=62"
    / "Runs"
    / "0002406475"
)

items = [i.name for i in run.glob("*")]

number = "123"
directory = "0002406475"
directory = "0002406475"

if (number == directory) and ("optim.pth" in items) and ("results.pth" in items):
    print("Skipping")
else:
    print("Running")
