"""Contracts for the curriculum import, simulation climate, and inference CLI.

These checks read source, so they run without PyTorch. One import check is
marked torch-dependent and skips unless CPU torch and the training stack are
installed.

    pytest -q tests/test_entrypoint_contracts.py
"""

import ast
import os
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNITY_PYTHON = os.path.join(ROOT, "unity", "Assets", "Python4Unity_Sample", "Python")

DEFAULT_MODEL = "TrainedModels/firefighter_policy_batch_0_20250804_143554.onnx"
DEFAULT_CONFIG = "TrainedModels/firefighter_policy_config_20250804_143517.json"
CLIMATE = {"temperature": 260.0, "humidity": 0.6, "precipitation": 600.0}
SIM_HELPERS = (
    "calibrate_simulation_params",
    "stress_test_simulations",
    "run_failed_simulation",
)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _parse(path):
    return ast.parse(_read(path))


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _assigned_climate_ctors(func):
    found = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if _call_name(node.value.func) == "ClimateConditions":
            found.append(node.value)
    return found


def _assert_climate_ctor(call, where):
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert set(CLIMATE) <= set(kwargs), f"{where} is missing a climate field"
    for name, expected in CLIMATE.items():
        got = _number(kwargs[name])
        assert got is not None and abs(got - expected) < 1e-9, f"{where} {name}={got}"


def _elo_paths():
    return [
        os.path.join(ROOT, "AI", "EloCurriculumManager.py"),
        os.path.join(UNITY_PYTHON, "AI", "EloCurriculumManager.py"),
    ]


def _sim_paths():
    return [
        os.path.join(ROOT, "QuicFire", "QuicFireSimulation.py"),
        os.path.join(UNITY_PYTHON, "QuicFire", "QuicFireSimulation.py"),
    ]


def _infer_paths():
    return [
        os.path.join(ROOT, "InferFirefighterOptionCritic.py"),
        os.path.join(UNITY_PYTHON, "InferFirefighterOptionCritic.py"),
    ]


def test_elo_imports_checkpoint_helpers_from_the_trainer():
    for path in _elo_paths():
        tree = _parse(path)
        trainer_names = set()
        agent_names = set()
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            names = {alias.name for alias in node.names}
            if node.module == "TrainFirefighterOptionCritic":
                trainer_names |= names
            if node.module == "AI.OptionCriticAgent":
                agent_names |= names
        assert {"save_config_to_file", "save_checkpoint"} <= trainer_names, path
        assert "save_config_to_file" not in agent_names, path
        assert "save_checkpoint" not in agent_names, path
        assert "OptionCriticAgent" in agent_names, path


def test_simulation_helpers_pass_the_standard_climate():
    for path in _sim_paths():
        functions = _functions(_parse(path))
        for name in SIM_HELPERS:
            func = functions[name]
            ctors = _assigned_climate_ctors(func)
            assert ctors, f"{path} {name} never builds ClimateConditions"
            for ctor in ctors:
                _assert_climate_ctor(ctor, f"{path} {name}")
            calls = [
                node for node in ast.walk(func)
                if isinstance(node, ast.Call) and _call_name(node.func) == "run_fire_simulation"
            ]
            assert calls, f"{path} {name} does not call run_fire_simulation"
            for call in calls:
                keywords = {kw.arg for kw in call.keywords}
                assert "module_rngs" not in keywords, f"{path} {name} still passes module_rngs"
                passed = False
                for kw in call.keywords:
                    if kw.arg == "climate":
                        passed = True
                if not passed:
                    # Positional climate is the third argument.
                    assert len(call.args) >= 3, f"{path} {name} omits climate"
                    climate_arg = call.args[2]
                    assert _call_name(climate_arg) in {None, "climate"} or isinstance(climate_arg, ast.Name)
                    assert isinstance(climate_arg, ast.Name) and climate_arg.id == "climate"
                    passed = True
                assert passed


def test_inference_flags_default_to_the_shipped_checkpoint_names():
    for path in _infer_paths():
        source = _read(path)
        tree = _parse(path)
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    constants[node.targets[0].id] = node.value.value
        assert constants["DEFAULT_MODEL_PATH"] == DEFAULT_MODEL
        assert constants["DEFAULT_CONFIG_PATH"] == DEFAULT_CONFIG

        parser = _functions(tree)["build_arg_parser"]
        flags = {}
        for node in ast.walk(parser):
            if not isinstance(node, ast.Call) or _call_name(node.func) != "add_argument":
                continue
            if not node.args:
                continue
            flag = node.args[0]
            if isinstance(flag, ast.Constant):
                default = {kw.arg: kw.value for kw in node.keywords}.get("default")
                flags[flag.value] = default
        assert isinstance(flags["--model"], ast.Name) and flags["--model"].id == "DEFAULT_MODEL_PATH"
        assert isinstance(flags["--config"], ast.Name) and flags["--config"].id == "DEFAULT_CONFIG_PATH"
        assert "Cannot start inference:" in source
        assert "pass --model and --config" in source


def test_missing_checkpoint_exits_before_loading_a_model():
    infer = os.path.join(ROOT, "InferFirefighterOptionCritic.py")
    script = textwrap.dedent(
        f"""
        import ast
        import os
        import sys
        tree = ast.parse(open({infer!r}, encoding="utf-8").read())
        keep = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {{
                "build_arg_parser", "checkpoint_path", "require_checkpoint"
            }}:
                keep.append(node)
            elif isinstance(node, ast.Assign):
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {{
                    "DEFAULT_MODEL_PATH", "DEFAULT_CONFIG_PATH"
                }}:
                    keep.append(node)
        module = ast.Module(body=keep, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {{"os": os, "sys": sys, "argparse": __import__("argparse")}}
        exec(compile(module, {infer!r}, "exec"), namespace)
        parser = namespace["build_arg_parser"]()
        defaults = parser.parse_args([])
        assert defaults.model == {DEFAULT_MODEL!r}
        assert defaults.config == {DEFAULT_CONFIG!r}
        chosen = parser.parse_args(["--model", "custom.onnx", "--config", "custom.json"])
        assert chosen.model == "custom.onnx"
        assert chosen.config == "custom.json"
        try:
            namespace["require_checkpoint"]("/no/such/ember-checkpoint.onnx", "model")
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise SystemExit("require_checkpoint returned for a missing file")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    # The missing-file path prints to stderr inside the child. Re-run the
    # message check by invoking the same function through a tiny driver that
    # we already executed; the child must have exited 0 only if the asserts
    # passed. Confirm the message with a direct call below.
    assert completed.returncode == 0, completed.stderr

    message = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import runpy
                import sys
                sys.argv = ["InferFirefighterOptionCritic.py"]
                # Importing the module needs torch. Check the source string instead.
                text = open("InferFirefighterOptionCritic.py", encoding="utf-8").read()
                assert "Cannot start inference: {{label}} not found at {{path}}" in text
                """
            ),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert message.returncode == 0, message.stderr


def test_missing_checkpoint_message_names_the_path(tmp_path=None):
    """require_checkpoint prints the missing path and exits 1, without torch."""
    infer = os.path.join(ROOT, "InferFirefighterOptionCritic.py")
    driver = textwrap.dedent(
        f"""
        import ast, os, sys
        tree = ast.parse(open({infer!r}, encoding="utf-8").read())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require_checkpoint")
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = {{"os": os, "sys": sys}}
        exec(compile(module, "require_checkpoint", "exec"), ns)
        ns["require_checkpoint"]("/missing/firefighter_policy.onnx", "model")
        """
    )
    completed = subprocess.run([sys.executable, "-c", driver], capture_output=True, text=True)
    assert completed.returncode == 1
    assert "Cannot start inference: model not found at /missing/firefighter_policy.onnx" in completed.stderr
    assert "--model and --config" in completed.stderr


def test_headset_main_walks_the_episode_and_root_takes_one_step():
    root_main = _read(os.path.join(ROOT, "InferFirefighterOptionCritic.py")).split('if __name__ == "__main__":', 1)[1]
    headset_main = _read(os.path.join(UNITY_PYTHON, "InferFirefighterOptionCritic.py")).split('if __name__ == "__main__":', 1)[1]
    assert "while not _done:" in headset_main
    assert "while not _done:" not in root_main
    assert "step_episode()" in root_main
    assert "require_checkpoint(model_path, \"model\")" in root_main
    assert "require_checkpoint(model_path, \"model\")" in headset_main


def test_checkpoint_helpers_are_the_trainer_functions():
    """torch-dependent: install CPU torch and requirements.txt to import the trainer."""
    try:
        import torch  # noqa: F401
        from TrainFirefighterOptionCritic import save_checkpoint, save_config_to_file
        import AI.EloCurriculumManager as elo
    except ImportError as exc:
        try:
            import pytest
        except ImportError:
            print(f"SKIP torch-dependent: {exc}")
            return
        pytest.skip(
            "torch-dependent: "
            f"{exc}. Install CPU torch and requirements.txt to run this import check."
        )
    assert elo.save_config_to_file is save_config_to_file
    assert elo.save_checkpoint is save_checkpoint
