#!/bin/bash

python -m timeit -s "import projects.swellex96_inv.data.bo.fwd as fwd" "fwd.main(simulate=True)"
python -m timeit -s "import projects.swellex96_inv.data.bo.fwd as fwd" "fwd.main(simulate=False)"
