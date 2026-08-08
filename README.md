# AI Grand Prix

Control and reinforcement-learning tools for Anduril's AI Grand Prix simulator.
The repository contains the work for two virtual qualifiers, with each qualifier
kept in its own self-contained directory.

## Repository layout

- [`vq1/`](vq1/) — vision-based control, Gymnasium environment, PPO training and evaluation.
- [`vq2/`](vq2/) — simulator contract probing and synchronized manual capture tools.

Each qualifier has its own `pyproject.toml`, `uv.lock`, scripts, controls, tests and
artifacts. Start with the README in the qualifier you want to run:

- [Virtual Qualifier 1 guide](vq1/README.md)
- [Virtual Qualifier 2 guide](vq2/README.md)

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- The official simulator for live MAVLink, video, control and training workflows

## Quick start

From the selected qualifier directory, install the locked environment:

```bash
uv sync
```

Run only the scripts documented for that qualifier, and follow their safety notes
before enabling live execution. Live tools use simulator network ports, so stop any
other client or probe that may already be connected.

## Safety

Passive probes are preferred when validating simulator connectivity. Active control
and training commands can move the vehicle; use the bounded probes and calibration
defaults described in the qualifier-specific documentation.

## License

No license is currently specified for this repository.
