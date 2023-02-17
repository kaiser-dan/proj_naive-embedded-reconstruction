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
  - [Repository Structure](#repository-structure)
  - [Reproducing experiments](#reproducing-experiments)
- [Documentation](#documentation)
- [Running the tests](#running-the-tests)
  - [Test Suite Organization](#test-suite-organization)
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
- [Conda \[4.14+\]](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

A complete list of packages is available in the `environment.yaml` file. Instructions for creating a controlled environment from this manifest is available below, in the [Installing](#installing) section.

## Installing

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the git-history and may need to be downloaded independently.
1. Open a terminal with Python and Conda installed and run the commands:
   ```
   $> conda env create -f environment.yaml
   $> conda activate EmbeddedNaive
   ```

This will install all necessary packages for you to be able to run the scripts and everything should work out of the box.

## Quick Start

[Describe makefile and reproduction script]

# Usage

[Usage guide]

## Repository Structure

[Describe structure]

## Reproducing experiments

[Describe experimental protocol docs]

# Documentation

[Describe documentation]


# Running the tests

All unit tests are written with [XXX test utility]().


## Test Suite Organization


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
details

## Acknowledgments
  - **Billie Thompson** - *Provided README and CONTRIBUTING template* -
  [PurpleBooth](https://github.com/PurpleBooth)
  - **George Datseris** - *Published workshop on scientific code; inspired organization for reproducibility* - [GoodScientificCodeWorkshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop)
