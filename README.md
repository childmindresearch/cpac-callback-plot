# `clplot` - callback.log Plot

[![Build](https://github.com/childmindresearch/cpac-callback-plot/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/cpac-callback-plot/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/cpac-callback-plot/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/cpac-callback-plot)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/childmindresearch/cpac-callback-plot/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/cpac-callback-plot)

## Installation

Install this package via :

```sh
pip install git+https://github.com/childmindresearch/cpac-callback-plot
```

## Quick start

```
usage: clplot [-h] [--output OUTPUT] [--overlap OVERLAP] [--label-min-percent LABEL_MIN_PERCENT] callback_log

Plot a timeline of pipeline events.

positional arguments:
  callback_log          Path to the callback.log file.

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to save the plot. If not provided, the plot will be shown in a window.
  --overlap OVERLAP     Allowed overlap between events in the same slot. Default: 5 seconds.
  --label-min-percent LABEL_MIN_PERCENT
                        Minimum percentage of the total timeline duration for an event to be labeled. Default: 0.01
```
