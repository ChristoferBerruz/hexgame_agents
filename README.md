# hexgame_agents
CS272 Hexgame agents for final assignment


## Getting started
This project adopts a poetry style tree structure where a python
packages must be inside a folder.

For dependency management, this repo uses poetry as a default.
You can install [poetry](https://python-poetry.org/docs/) and simply
run

```code
poetry install
```

This will create a poetry environment with all dependencies. You can also
manually install the following dependencies:

```code
pip install "pettingzoo[classic]"
pip install torch
```

## Hexgame as a submodule
This project uses submodules to pull updates from the shared
ourhexgame repo. To update the submodules, simply do

```
git pull --recurse-submodules
```