# Configuration, observations, and rewards

The trainer reads `QuicFireConstants.BASE_CONFIG`. The sections below are the grid, the observation channels, the action row, and the containment reward. How to run a simulation, a training episode, or inference is in the [README](../README.md#how-to-run).

## Configuration

| Key | Default | Role |
| --- | --- | --- |
| `nx`, `ny`, `nz` | 100, 100, 50 | Simulation grid |
| `dx`, `dy` | 2 m | Horizontal cell size |
| `max_grid_height` | 300 m | Top of the vertical grid |
| `max_elevation` | 50 m | Terrain generator cap |
| `total_time` | 10000 s | Episode horizon |
| `num_episodes` | 1 | Trainer episodes |
| `obs_grid_x`, `obs_grid_y` | 20, 20 | Agent observation grid |
| `num_agents` | 4 | Firefighters on the shared policy |
| `random_seed` | 12243261 | Master seed |

## Observation Space

The observation is a float32 box of shape `(obs_grid_x, obs_grid_y, 9)`, default `(20, 20, 9)`. The network permutes that to channels-first before the convolutions.

| Channel | Name |
| --- | --- |
| 0 | `heat_flux` |
| 1 | `energy_packets` |
| 2 | `temperature` |
| 3 | `fuel_density` |
| 4 | `moisture_fraction` |
| 5 | `wind_magnitude` |
| 6 | `agent_positions` |
| 7 | `burn_box` |
| 8 | `placeholder` |

## Action Space

`select_and_execute_options` returns one row per agent: `[agent_idx, action_type, x, y]`, with `x` and `y` on the observation grid. The environment consumes that array in `step`.

Agents spend finite water, retardant, and air-support charges. A firebreak has no resource cost. Ranges and capacities are the `WATER_*`, `RETARDANT_*`, `AIR_SUPPORT_*`, and `FIREBREAK_*` constants in `QuicFireConstants.py`.

## Reward

The score is a prescribed-burn containment objective. Fuel burned inside the burn box is rewarded. Active fire and fuel loss outside the box are penalized, along with water, retardant, air support, time, invalid actions, and movement. A contained mission and a fire that escapes or times out use the mission constants in `QuicFireConstants`. Per-agent reward is blended with the global reward by `reward_alpha` (default 0.8).
