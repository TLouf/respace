<div align="center">

# ReSpace

[![PyPI](https://img.shields.io/pypi/v/respace.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/respace.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/respace)][pypi status]
[![License](https://img.shields.io/pypi/l/respace)][license file]

[![Read the documentation at https://respace.readthedocs.io/](https://img.shields.io/readthedocs/respace/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/TLouf/respace/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/TLouf/respace/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)][ruff]

[pypi status]: https://pypi.org/project/respace/
[license file]: https://github.com/TLouf/respace/blob/main/LICENSE
[read the docs]: https://respace.readthedocs.io/
[tests]: https://github.com/TLouf/respace/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/TLouf/respace
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[ruff]: https://github.com/charliermarsh/ruff

</div>

ReSpace is a Python package that allows you to keep track of the results of different
computations along with the parameter values that generated them. You specify their
names, the function that generates them, the name and default values of the parameters
they depend on and you're good to go: no more trying to remember what parameters this
value was computed for, building dictionaries of dictionaries (of dictionaries) to
store them, or generally worrying about these things.


## Features

- Compute and store some result, indexing it based on the values of the parameters
  it was computed for.
- Compute results that depend on others reliably and easily.
- Retrieve a previously computed result for a set of parameters.
- Add new parameters seemlessly.
- Handle parameter defaults.
- Save your results at different paths depending on the set of parameters they were
  computed for, with little to no effort.
- Track how long each computation takes to identify bottlenecks in your pipeline.


## Installation

You can install the latest release of ReSpace via [pip] from [PyPI]:

```console
$ pip install respace
```

Or, if you want the latest development version from GitHub with [git]:

```console
$ pip install git+https://github.com/TLouf/respace
```

ReSpace requires Python 3.8+ and depends on the [xarray] library.

[git]: https://git-scm.com/
[pip]: https://pip.pypa.io/
[pypi]: https://pypi.org/
[xarray]: https://docs.xarray.dev/en/stable/

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license page],
_ReSpace_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from a fork of [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

<!-- urls -->
[@cjolowicz]: https://github.com/cjolowicz
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/TLouf/respace/issues

<!-- github-only -->
[license page]: https://github.com/TLouf/respace/blob/main/LICENSE
[contributor guide]: https://github.com/TLouf/respace/blob/main/CONTRIBUTING.md
