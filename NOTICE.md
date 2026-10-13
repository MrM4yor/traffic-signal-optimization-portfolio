# License scope and attribution

Original project code and documentation in this repository are licensed under
MIT; see [LICENSE](LICENSE). The attribution name is the author's public GitHub
handle, MrM4yor. AI tools assisted implementation, debugging, review, and writing.

The MIT license does not change the licenses of dependencies or grant rights in
external datasets, networks, trademarks, or third-party software. This repository
does not bundle those dependency implementations, raw traffic exports, network
assets, or trained model files. `evidence/` contains generated aggregate research
summaries, not a redistribution of the underlying ATSPM dataset.

## Research inputs and tools

- [Eclipse SUMO](https://sumo.dlr.de/docs/) supplied microscopic traffic simulation
  and the TraCI/libsumo interfaces used by the main research project.
- [UDOT ATSPM](https://udottraffic.utah.gov/ATSPM/) supplied public controller and
  detector observations used in historical reconstruction for March 19, 2026.
  Public availability is not treated here as a license to redistribute raw data.
- [RESCO](https://github.com/Pi-Star-Lab/RESCO) supplied the network basis used in
  the main reconstruction experiments. Its repository is GPL-3.0 licensed; no
  RESCO network or implementation is bundled in this portfolio.
- [scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/),
  [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org/), and
  [joblib](https://joblib.readthedocs.io/) supply external modeling/numerical tools.
  Their own licenses apply when installed.

RESCO citation: James Ault and Guni Sharon (2021), *Reinforcement Learning
Benchmarks for Traffic Signal Control*, NeurIPS Datasets and Benchmarks Track.

Derived measurements are historical simulation/reconstruction results, not
independent proof of real-world traffic improvements or ground-truth identities.
The project is not endorsed by the organizations listed above.
