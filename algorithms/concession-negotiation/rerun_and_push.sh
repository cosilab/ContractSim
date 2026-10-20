#!/bin/bash
# sweep + eval + plots -> paper images -> compile -> push to Overleaf
set -ex
R=/home/bhavyesh/ContractSim
date
cd $R/algorithms/concession-negotiation
./run_sweep_eval.sh
test sweep/results.csv -nt sweep/sweep_env5.log
date
$R/.venv/bin/python $R/paper-stuff/plots/scripts/pareto-frontier-panel.py pareto_frontiers_baseline
cp $R/paper-stuff/plots/outputs/pareto_frontiers_baseline/{env-0?_pareto.pdf,pareto_panel_env-01-03-05.pdf} $R/Paper/images/pareto-frontiers/
cd $R/Paper
latexmk -pdf -interaction=nonstopmode main.tex > /dev/null || true
latexmk -pdf -interaction=nonstopmode appendix.tex > /dev/null || true
grep -a '^!' main.log appendix.log && exit 1 || true
git add images/pareto-frontiers
git commit -m "Regenerate concession baseline Pareto plots"
git pull --rebase
git push
latexmk -pdf -interaction=nonstopmode main.tex > /dev/null || true
latexmk -pdf -interaction=nonstopmode appendix.tex > /dev/null || true
date
echo ALL DONE
