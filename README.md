This is the code of the paper "On the Limitations of Elo: Real-World Games, are Transitive, not Additive" accepted in AISTATS 2023 (https://arxiv.org/abs/2206.12301).

First, run `pip install -e .` in this folder to install the package and its dependency.

Then one can launch 2 scripts which run in minutes:
- `ipython -i examples/plot_elo_breaks.py` to produce Figure 2 of the paper "Elo score fails to rank players for a transitive game"
- `ipython -i examples/plot_disc_rating_works.py` to produce Figure 3 of the paper "Extended Elo score manages to rank players for a transitive game"

Code to produce Figure 6 and 7 is provided, although running these experiments might take time (hours).
For Figure 6 (prediction performances on spinning top data) one should go in the `expes/spinning_top` folder, and successively run:
- `main_pred.py`
- `print_results.py`
- `figure_pred.py`

For Figure 7 one should go to the `expes/chess_starcraft` folder and successively run:
- `main_chess.py`
- `main_starcraft.py`
- `figure_chess_starcraft.py`

If you found this code useful, please cite

@article{bertrand2022limitations,
  title={On the Limitations of Elo: Real-World Games, are Transitive, not Additive},
  author={Bertrand, Quentin and Czarnecki, Wojciech Marian and Gidel, Gauthier},
  journal={arXiv preprint arXiv:2206.12301},
  year={2022}
}
