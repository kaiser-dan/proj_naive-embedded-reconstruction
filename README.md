# Multiplex Reconstruction via Graph Embeddings

This project provides source code for the reconstruction of multiplex networks from partial structural observations leveraging graph embeddings. The repository also contains the original scientific analyses developed by the Authors (see below) for the paper

- _In preparation_

# Contents

- [Multiplex Reconstruction via Graph Embeddings](#multiplex-reconstruction-via-graph-embeddings)
- [Contents](#contents)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [Reproducing experiments](#reproducing-experiments)
  - [Package Structure](#package-structure)
- [Documentation](#documentation)
- [Tests](#tests)
- [Other Information](#other-information)
  - [Built With](#built-with)
  - [Contributing](#contributing)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)


# Getting Started

The code base for this project is written in Python with package management handled with Conda.

These instructions will give you a copy of the project up and running on
your local machine for development, testing, and analysis purposes.

## Prerequisites

A compatible Python install is needed to begin - the package management is handled by Conda as described below.
- [Python \[3.10+\]](https://python.org/downloads/)
- [GNU Make \[4.2+\]](https://www.gnu.org/software/make/) (only needed for our provided `makefile` - see [Reproducing Experiments](#reproducing-experiments))

A complete list of utilized packages is available in the `requirements.txt` file. There is, however, a package dependency hierarchy where some packages in the `requirements.txt` are not strictly necessary for the utilization of package infrastructure. The core requirements are listed as dependencies in the build instructions. Further instructions for creating a controlled environment from this manifest is available below, in the [Installing](#installing) section.


## Installing

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the git-history and may need to be downloaded independently - see [Reproducing Experiments](#reproducing-experiments) for more information.
    ```bash
    git clone --depth 1 GITADDRHERE
    ```
2. (Optional) Open a terminal with Python installed and create a new virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the package
   ```bash
   pip install .
   ```
   > **NOTE:** We recommend you use the provided `makefile` to handle installations and automatic testing. You can install the package, testing and plotting packages, and run the package tests all with the default `make` target, i.e., by running `make`.
4. (Optional) If you wish to reproduce the experiments (and not did run the `make` command), install the additional package dependencies to use Snakemake
   ```bash
   pip install .[workflow]
   ```


This will install all necessary packages for you to be able to run the scripts and everything should work out of the box.

## Quick Start

**CREATE A SMALL LIL BABY EXAMPLE**

# Usage

[Usage guide]

## Reproducing experiments

In the interest of reproducibility and scientific rigor, we have prepared a `makefile` that will reproduce the main analyses present in the accompanying manuscript. Broadly, this `makefile` will do the following:
1. Setup python environment.
2. Retrieve and prepare the relevant multiplex datasets from online archives
3. Run the main experiments on these data and save the results to disk
4. Prepare the figures as they appear in the manuscript.

This `makefile` also contains rules for cleaning downloaded and temporary files as well as retrieving binaries for sampling LFR benchmarks.

To use this `makefile`, which requires `GNUMake`, simply run
  ```bash
  make
  ```
to install the necessary packages to use our source code.

Reproducing our experiments on real datasets can be done with
  ```bash
  make reproduce
  ```

Reproducing our experiments on synthetic datasets can be done with
  ```
  snakemake --cores [num_cores] --configfile workflow/configurations/all.yaml
  ```

> **NOTE:** Running all synthetic experiments in one jobset, as above, is _extremely_ computationally expensive. We originally ran our results in chunks and on different machines simultaneously. This can be done using the provided `workflow/configurations/ex[##].yaml` files. Simply swap out `all.yaml` with `ex[##].yaml` in the same snakemake command above.


## Package Structure

[Fill in here]

# Documentation

This repository does not maintain extensive independent documentation for its source code. We do, however, include documentation and notes on scientific experiments we've conducted throughout the project. If you are interested in seeing these notes, please email [Daniel Kaiser](mailto:kaiserd@iu.edu) with your inquiry.

We have, however, kept all experimental protocols related to the final experimental designs of the published results in this public repository. These can be found in `docs/` with the appropriate names matching the results as presented in the manuscript.


# Tests

All unit tests are written with [pytest](docs.pytest.org).

Tests can be run directly with the commands:
```bash
pip install .[test]
pytest tests/
```

As above, we recommend making use of the included `makefile` to handle installations and testing. The default target will ensure all dependencies are up to date and tests are reran.
```bash
make
```

Alternatively, the tests alone can be conducted with
```bash
make check
```


# Other Information
## Built With
  - [ChooseALicense](https://choosealicense.com/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/kaiser-dan/proj_sable-spin-duplexes/tags).

## Authors

All correspondence shoulld be directed to [Daniel Kaiser](mailto:kaiserd@iu.edu).

- Daniel Kaiser
- Siddharth Patwardhan
- Minsuk Kim
- Fillippo Radicchi

## License

This project is licensed under the [MIT License](LICENSE.md)
Creative Commons License - see the [LICENSE](LICENSE.md) file for
details.

## Acknowledgments
  - **Billie Thompson** - *Provided README and CONTRIBUTING template* -
  [PurpleBooth](https://github.com/PurpleBooth)
  - **George Datseris** - *Published workshop on scientific code; inspired organization for reproducibility* - [GoodScientificCodeWorkshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop)
