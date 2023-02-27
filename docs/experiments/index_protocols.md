# Experimental Protocol Index

Embedding-based Multiplex Reconstruction experiment index.

This document maintains indexing data for experiments asociated to this project. Short summaries of the projects are given here, however, full details on each experiment are available in the associated protocols. These protocols are prefaced with `protocol_` and in the same relative directory as this index.

## Contents

- [Experimental Protocol Index](#experimental-protocol-index)
  - [Contents](#contents)
- [Operative Experiments](#operative-experiments)
  - [Active](#active)
  - [Completed](#completed)
  - [Planned](#planned)
- [Inoperative Experiments](#inoperative-experiments)
  - [Failed Experiments](#failed-experiments)
  - [Depreciated experiments](#depreciated-experiments)


# Operative Experiments

Experiments are considered "operative" if they are actively being developed or ran, being designed, or completed without problems.

## Active

Active experiments are settings currently being developed or analyzed.

- ex27
  - Associated experiments: ex26, ex28, ex29
  - Summary: N2V with configuration-likelihood degree feature alone

- ex28
  - Associated experiments: ex26, ex27, ex29
  - Summary: N2V with configuration-likelihood degree and embedding distance features


## Completed

Completed experiments are settings with generated, analyzed results that are without question.

- ex19
  - Date completed: X
  - Associated experiments: ex20
  - Summary: X
- ex20
   Date completed: X
  - Associated experiments: ex19
  - Summary: X
- ex21
  - Date completed: X
  - Associated experiments: ex22
  - Summary: X
- ex22
  - Date completed: X
  - Associated experiments: ex21
  - Summary: X
- ex23-verify
  - Date completed: 2023-02-22
  - Associated experiments: ex24-verify, ex25-verify
  - Summary: N2V distance-based likelihood reconstruction
- ex24-verify
  - Date completed: 2023-02-22
  - Associated experiments: ex23-verify, ex25-verify, ex26
  - Summary: N2V logistic regression with component-biases
- ex25-verify
  - Date completed: 2023-02-22
  - Associated experiments: ex23-verify, ex24-verify, ex26
  - Summary: N2V logistic regression
- ex26
  - Date completed: 2023-02-22
  - Associated experiments: ex23-verify, ex24-verify, ex25-verify
  - Summary: N2V logistic regression with expanded feature set

## Planned

Planned experiments are experiments that are planned or actively being designed, but before active development starts.

- ex29
  - Associated experiments: ex26, ex27, ex28
  - Summary: N2V with configuration-likelihood degree, embedding distance features and component biases


# Inoperative Experiments

Experiments are considered "inoperative" if their results should be considered flawed/incorrect.

## Failed Experiments

Failed experiments are experimental setups that are bugged or incomplete. They cannot be reproduced as is.

- ex04
- ex05
- ex06
- ex07

## Depreciated experiments

Depreciated experiments may, unlike failed experiments, be reproduced as is but the methodolofgies and theories behind their implementation may be unreasonable or nonsensical.

- ex07R
- ex08R
- ex09
- ex10
- ex11
- ex12
- ex17