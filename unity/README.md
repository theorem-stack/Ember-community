# Ember

Ember is a VR wildfire firefighter trainer for Meta Quest 3, built in 5 weeks for a SERDP wildfire training problem. This repository is the project's own scripts. The Unity scene, art, and third-party packages are not included.

## Architecture

The fire spread model is Python under `Assets/Python4Unity_Sample/Python`. On the headset it runs through an embedded CPython interpreter. An option-critic agent trains off the headset with an Elo curriculum. The headset loads an ONNX policy and runs inference. The GPU side keeps a 2D burn map and a layered 2.5D fluid field that advects smoke.

The ONNX weights are not in this snapshot. `PythonInferenceBridge` looks for `TrainedModels/firefighter_policy_batch_0_20250804_143554.onnx` next to the included policy config JSON.

## Layout

| Path | Role |
| --- | --- |
| `Assets/Python4Unity_Sample/Python/QuicFire/` | Spread engine: fuel, embers, wind, ignition, burn box, packet plumes, `QuicFireTuning` |
| `Assets/Python4Unity_Sample/Python/AI/` | Option-critic network and agent, firefighter environment, Elo curriculum, D* Lite paths |
| `Assets/Python4Unity_Sample/Python/TrainFirefighterOptionCritic.py` | Training entry point |
| `Assets/Python4Unity_Sample/Python/InferFirefighterOptionCritic.py` | Headset inference entry point |
| `Assets/Python4Unity_Sample/Python/TrainedModels/` | Policy config JSON the inference script reads. Weights are not included. |
| `Assets/Python4Unity_Sample/Scripts/PythonInferenceBridge.cs` | Unity side of inference. Calls the Python module through the Chaquopy4Unity API. |
| `Assets/Python4Unity_Sample/Scripts/BurnMapManager.cs` | Uploads fires into `BurnMap.compute` |
| `Assets/Custom/BurnMap.compute` | 2D burn-map splat |
| `Assets/Resources/HeadsetFluid.compute` | Layered 2.5D fluid: advection, viscosity, buoyancy, wind, pressure projection |
| `Assets/Python4Unity_Sample/Scripts/HeadsetFluidSimulation.cs` | Dispatches the fluid and draws plume billboards |
| `Assets/Resources/PlumeFluidParticles.shader` | Billboard shader for those particles |
| `Assets/Python4Unity_Sample/Scripts/PlumRenderer.cs` | CPU plume particles, used when the fluid is off |
| `Assets/Custom/VegetationBurnShader.shader` | Vegetation dissolve from the burn map |
| `Assets/Custom/UnlitGridUI.shader` | Mission-control grid |
| `tests/test_quicfire_tuning.py` | Checks that tuning overrides keep the default constants |
| `Packages/manifest.json` | Unity packages this project was built against |

Python imports use `QuicFire` and `AI` as top-level packages. Put `Assets/Python4Unity_Sample/Python` on `PYTHONPATH`. From the repo root:

```bash
python3 tests/test_quicfire_tuning.py
```

The private project pinned `numpy==1.26.4`, `torch==2.6.0`, `scipy==1.15.3`, `scikit-image==0.25.2`, `gymnasium==1.2.0`, `onnx==1.18.0`, and `onnxruntime==1.22.1`. Debug plots also import matplotlib.

## Rebuilding the Unity app

Open the scripts in Unity 2022.3.17f1. `Packages/manifest.json` is the dependency list from the private project. The packages the scripts call for directly are:

- Meta XR SDK: `com.meta.xr.sdk.core` 77.0.0, `com.meta.xr.sdk.interaction.ovr` 77.0.0, `com.meta.xr.simulator` 77.0.0, `com.meta.xr.runtimeoptimizer` 0.1.0
- `com.unity.xr.openxr` 1.9.1, `com.unity.xr.management` 4.5.1, `com.unity.xr.interaction.toolkit` 2.5.4
- `com.unity.scripting.python` 7.0.1 for the editor interpreter
- `com.unity.textmeshpro` 3.0.6 (`UIAgentResourceEntry`, `UIAgentResourceDisplay`, and `UIMissionLogger` use TextMeshPro)
- Unity UI (`com.unity.ugui`) and the Input System, which the XR packages bring in for `XRCameraController`

The Android build also needs [Chaquopy4Unity](https://github.com/NON906/python4unity) (Chaquopy 15.0.1). That plugin is not in this snapshot. `PythonInferenceBridge` calls `PythonUtil` from it. Copy the plugin in, point its Python site-packages at `Assets/Python4Unity_Sample/Python`, and add a scene that wires the bridge, burn map, fluid, and renderers. Scenes, prefabs, materials, and textures are not included.

The fluid step reads a burn-map render texture and the plume packet (`origin`, wind `plume_u/v/w`, `total_heat`, `h_sub`, `ell`, `k`). With **Simulate** on, plume particles follow the velocity field. The burn-map kernel is separate and still only splats fire intensity, height, and burn progress.
