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
- [GNU Make \[4.2+\]](https://www.gnu.org/software/make/) (only needed for our provided `makefile` - see [Reproducing Experiments](#reproducing-experiments))

A complete list of packages is available in the `environment.yaml` file. Instructions for creating a controlled environment from this manifest is available below, in the [Installing](#installing) section.

> _Note: We personally recommend using mambaforge, an extension of conda that is considerably faster and more robust. Further information can be found in the [Mamba docs](https://mamba.readthedocs.io/en/latest/index.html)_.

## Installing

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the git-history and may need to be downloaded independently - see [Reproducing Experiments](#reproducing-experiments) for more information.
2. Open a terminal with Python and Conda installed and run the commands:
   ```
   $> conda env create -f environment.yaml
   $> conda activate EmbeddedNaive
   ```

This will install all necessary packages for you to be able to run the scripts and everything should work out of the box. 

## Quick Start

In the interest of reproducibility and scientific rigor, we have prepared a `makefile` that will reproduce the main analyses present in the accompanying manuscript. Broadly, this `makefile` will do the following:
1. Setup python environment.
2. Retrieve and prepare the relevant multiplex datasets from online archives
3. Run the main experiments on these data and save the results to disk
4. Prepare the figures as they appear in the manuscript.

This `makefile` also contains rules for cleaning downloaded and temporary files as well as retrieving binaries for sampling LFR benchmarks.

To use this `makefile`, which requires `GNUMake`, simply run
  ```
  $> make
  ```
to install the necessary packages to use our source code.

Reproducing our experiments can be done with
  ```
  $> make reproduce
  ```


Finally, cleaning all downloaded/generated files can be accomplished with
  ```
  $> make clean
  ```

# Usage

[Usage guide]

## Repository Structure

[Fill in here]

## Reproducing experiments

As mentioned above, the experiments can be reproduced in their entirety with the command
```
$> make reproduce
```

This will reproduce all analyses present within the manuscript within their respective order. It will take a fair bit of time as many multiplexes are generated, embedded, reconstructed, and compared throughout. It is generally recommended you do _not_ run this command directly.

Each figure in the manuscript has a corresponding script which will reproduce _only_ the experiments necessary to create that figure (and actually do the figure synthesis and save to disk as well). These are located in the `script/` directory and it is highly recommended that you utilize these scripts if you only want to reproduce a portion of the manuscript. Additional information can be found in `scripts/REPRODUCING_RESULTS.md`

# Documentation

This repository does not maintain extensive independent documentation for its source code. We do, however, include documentation and notes on scientific experiments we've conducted throughout the project. If you are interested in seeing these notes, please email [Daniel Kaiser](mailto:kaiserd@iu.edu) with your inquiry.

We have, however, kept all experimental protocols related to the final experimental designs of the published results in this public repository. These can be found in `docs/experiments/` with the appropriate names matching the results as presented in the manuscript.

<!-- Additionally, a copy of individual derivations can be found in `docs/` that are highly suggestive of methodological choices and implications for our work. -->

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
details.

## Acknowledgments
  - **Billie Thompson** - *Provided README and CONTRIBUTING template* -
  [PurpleBooth](https://github.com/PurpleBooth)
  - **George Datseris** - *Published workshop on scientific code; inspired organization for reproducibility* - [GoodScientificCodeWorkshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop)
