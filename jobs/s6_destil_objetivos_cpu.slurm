#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-destil-obj
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# D-destil: objetivos de S6 = 0.5 · JEV + 0.5 · Raptor (maestro de Owen, solo lectura) -> S6F0/data/targets_s6f0_destil.csv
set -euo pipefail
PY=/lustre/proyectos/p082/venvs/rsna/bin/python
"$PY" -I /lustre/proyectos/p082/runs_joako/exp_s6_semillas/herramientas/destilar_objetivos.py --alfa 0.5
