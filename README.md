# Traffic Signal Optimization and Historical Reconstruction

Personal research project by **MrM4yor**, developed with AI assistance. This case
study combines machine learning, simulation engineering, and the reconstruction
of traffic from anonymous signal-controller observations.

**Status:** research prototype with historical benchmark evidence and an offline
demo. No real-world deployment or universal performance improvement is claimed.

## Two connected problems

**Control:** Can a signal anticipate stops and braking, then choose a better legal
phase? I explored evolutionary policies, cached-data neural predictors, and
same-state counterfactual phase ranking across synthetic network families.

**Reconstruction:** Can historical ATSPM detector pulses and recorded signal
transitions support a physically plausible SUMO replay? I developed an auditable
pipeline for two intersections on 400 South in Salt Lake City, distinguishing
measured signals from inferred traffic and driver behavior.

## Selected results and tradeoffs

Historical 20-route comparisons against the existing topology-specific controller
portfolio use positive percentages for reductions in the metric:

| Controller profile | Stops | Waiting | Time loss | CO2 |
| --- | ---: | ---: | ---: | ---: |
| Balanced neural predictor | 2.96% fewer | 1.65% less | 1.68% less | 0.61% less |
| Stop-first predictor + ranker | 9.95% fewer | **5.70% more** | **0.13% more** | 1.02% less |

A separate 20-route set showed the MLP producing **2.68% more stops and 3.15% more
waiting**. The results show a useful speed/quality tradeoff and route sensitivity,
not an all-scenario replacement. Reviewed code changes have not been rebenchmarked;
these numbers describe the original experiments.

The accepted reconstruction matched 1,850/1,850 boundary events with 0.228054 s
timing MAE. Internal coverage was 903 matches from 932 observations, with 24.4495 s
matched timing MAE. A newer trial improved internal timing but failed the boundary
tolerance and was not promoted. Both runs recorded all 1,884 vehicles arriving,
zero collisions/teleports, and no warning output.

The conditional forward surrogate reported 91.86% exact detector-sequence accuracy
on held-out vehicle identities. It needs future surrounding traffic context and
is not an autonomous forecast. Repeated replays of one fleet are not independent
days of real traffic.

## Try the offline demo

Python 3.14 was tested. The results demo uses only the standard library:

```sh
python demo.py --check
python demo.py
```

The second command writes `.demo/REPORT.md` and `.demo/report.html`. It verifies
export hashes, recomputes controller reductions from preserved totals, and shows
the accepted-versus-rejected reconstruction. **It does not run SUMO or train a
model.** See [results and interpretation](docs/RESULTS.md).

For the code examples and unit tests:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Code to inspect

| Example | Engineering idea |
| --- | --- |
| [Event matching](code/event_matching.py) | Maximum coverage, then minimum timing error; duplicate timestamps remain distinct events |
| [Evolutionary policy](code/evolution_policy.py) | Fixed-scale inputs, bounded residual actions, and neuron-aligned crossover |
| [Neural phase ranker](code/train_neural_phase_ranker.py) | Reusable-data fitting with separate route-group validation and final test sets |
| [Predictor blending](code/blended_predictor.py) | Class-aligned probability blending and compatibility checks |
| [Sensor adapter](code/sensor_features.py) | A fixed-location feature interface separating sensing from learned control |
| [ATSPM parsing](code/atspm_common.py) | Streaming timestamp/event parsing and metadata helpers |

`reference/` contains annotated full-pipeline implementations for inspection.
They require the main project's datasets, modules, and SUMO setup and are not demo
entry points. [Architecture and methodology](docs/ARCHITECTURE.md) explains them.

## Reproducibility and authorship

[PROVENANCE.json](PROVENANCE.json) identifies the main-repository revision, copied
code hashes, original evidence hashes, and the aggregate transformations. A
maintained exporter updates this subset while protecting handwritten portfolio
files. The main repo is private, so employers can assess the code/evidence included
here without needing access to it.

AI tools assisted implementation, debugging, review, and documentation. This is
presented as an ongoing personal research project, with unresolved limitations
visible. Original code and documentation are open source under the [MIT license](LICENSE).
Dependency and research-input attribution is in [NOTICE.md](NOTICE.md); underlying
datasets and network assets are not redistributed or relicensed here. See the
[publication review](docs/PUBLICATION.md) for the privacy audit and release scope.
