---
hide-toc: true  # remove RHS sidebar
---

<div style="text-align: center;">

# octavius: The Next Generation Simulation Analysis Toolkit

<div style="margin-bottom: 1.5em;"></div>

[![PyPI](https://img.shields.io/pypi/v/octavius)](https://pypi.org/project/octavius/)
[![DOI](https://zenodo.org/badge/1136349333.svg)](https://doi.org/10.5281/zenodo.22166418)
[![Python](https://img.shields.io/pypi/pyversions/octavius)](https://pypi.org/project/octavius/)
[![CI](https://github.com/jp-duminy/octavius/actions/workflows/ci.yml/badge.svg)](https://github.com/jp-duminy/octavius/actions)
[![codecov](https://codecov.io/gh/jp-duminy/octavius/graph/badge.svg?token=71Y900J2OY)](https://codecov.io/gh/jp-duminy/octavius)
[![Licence](https://img.shields.io/badge/licence-BSD--3--Clause-blue)](https://github.com/jp-duminy/octavius/blob/main/LICENCE)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

</div>

<div style="margin-bottom: 3.0em;"></div>

```{image} _static/banner.webp
:alt: Octavius
:align: center
:width: 700px
```

<div style="margin-bottom: 3.0em;"></div>
<div style="text-align: center;">

**Version:** 0.9.2.1

**Useful Links:** | [Installation](getting_started/installation.md) | [Quickstart](getting_started/quickstart.md) | [Five-Minute Guide](getting_started/five_minute_guide.md) | [GitHub](https://github.com/jp-duminy/octavius) | [What's New](https://github.com/jp-duminy/octavius/releases)

</div>

<div class="landing-body" markdown="1">

`octavius` is a high-performance, fully-parallelised galaxy simulation analysis toolkit written entirely in Python. When run on a simulation snapshot, `octavius` produces HDF5 analysis catalogues containing properties and membership information for haloes and galaxies. 

Features include:

- Support for [SWIFT](https://swift.strw.leidenuniv.nl/) (EAGLE, KIARA, COLIBRE), [SIMBA](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2827D/abstract), and [TNG](https://www.tng-project.org/) snapshots
- Support for [AHF](https://iopscience.iop.org/article/10.1088/0067-0049/182/2/608), [HBT-HERONS](https://hbt-herons.strw.leidenuniv.nl/) and [SUBFIND](https://www.tng-project.org/data/docs/specifications/#sec2b) halo catalogues
- Snapshot-agnostic catalogues
- Built-in galaxy finding with a 6D friends-of-friends algorithm
- Computes over fifty properties for haloes and galaxies (including subhaloes)
- Photometry in all [FSPS](https://dfm.io/python-fsps/current/)-compatible bands with dust attenuation (no radiative transfer)
- User-friendly API for working with output catalogues
- Comprehensive membership mapping, including hierarchies
- Standalone analysis tools (on-the-fly pipeline stages)
- Comprehensive unit and regression tests

To get started, please refer to the [installation](getting_started/installation.md) guide; for a brief overview of the package, please see the [five-minute guide](getting_started/five_minute_guide.md).

Octavius is the spiritual successor to [caesar](https://caesar.readthedocs.io/). `caesar` users should please refer to the [Caesar users guide](guide/caesar_users_guide.md).

</div>

<div style="margin-bottom: 3.0em;"></div>

<div style="text-align: center;">
<pre style="display: inline-block; text-align: left;">
████████████████████████████████████
▄                                  ▄
▄ ░█▀█░█▀▀░▀█▀░█▀█░█░█░▀█▀░█░█░█▀▀ ▄
▄ ░█░█░█░░░░█░░█▀█░▀▄▀░░█░░█░█░▀▀█ ▄
▄ ░▀▀▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀▀▀░▀▀▀ ▄
▄                                  ▄
████████████████████████████████████
</pre>
</div>

```{toctree}
:hidden:
:caption: Getting Started

getting_started/index
```

```{toctree}
:hidden:
:caption: User Manual

guide/index
```

```{toctree}
:hidden:
:caption: Features

features/index
```

```{toctree}
:hidden:
:caption: Examples

examples/index
```

```{toctree}
:hidden:
:caption: API Reference

api/index
```

```{toctree}
:hidden:
:caption: Developer Manual

developers/index
```