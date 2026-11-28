#############################################################
#   FILE NAME: test_main.py
#
#   DESCRIPTION: This file contains unit tests for the
#   command line in main.py (play and bench). They live
#   outside unit_test.py so they run on machines without
#   tkinter; the play test swaps in a fake GUI module.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import sys
import types

import pytest
import torch
from PIL import Image

#############################################################
#   Files Import
#############################################################

import main as cli
from Library.model import create_model, save_checkpoint

#############################################################
#   FUNCTIONS
#############################################################

def make_split(root, classes=("bus", "hydrant", "bicycle")):
    """
    FUNCTION NAME: make_split

    DESCRIPTION: Writes a tiny split folder with two
    solid-color images per class.

    INPUTS: root (Path), classes (tuple of str)

    OUTPUTS: root (Path)
    """
    for number, label in enumerate(classes):
        (root / label).mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (20, 20), (number * 80, 0, 0)).save(root / label / f"{index}.png")
    return root

#############################################################
#   CLASSES
#############################################################

class TestCommandLine:
    """
    CLASS NAME: TestCommandLine

    DESCRIPTION: This class contains unit tests for main.py.
    """

    def test_no_command_prints_help(self, capsys):
        """
        FUNCTION NAME: test_no_command_prints_help

        DESCRIPTION: With no command, main prints the help and
        returns 0.

        INPUTS: capsys (pytest fixture)

        OUTPUTS: None
        """
        assert cli.main([]) == 0
        output = capsys.readouterr().out
        assert "bench" in output and "play" in output

    def test_bench_synthetic(self, capsys):
        """
        FUNCTION NAME: test_bench_synthetic

        DESCRIPTION: The default bench runs oracle, random and
        noisy on synthetic tiles, and the oracle scores 100%.

        INPUTS: capsys (pytest fixture)

        OUTPUTS: None
        """
        assert cli.main(["bench", "-n", "20"]) == 0
        output = capsys.readouterr().out
        assert "synthetic" in output
        oracle_row = next(line for line in output.splitlines() if line.startswith("oracle"))
        assert "100.0%" in oracle_row
        assert any(line.startswith("random") for line in output.splitlines())
        assert any(line.startswith("noisy") for line in output.splitlines())

    def test_bench_folder_with_model(self, tmp_path, capsys):
        """
        FUNCTION NAME: test_bench_folder_with_model

        DESCRIPTION: bench reads a split folder and runs a
        model checkpoint on the CPU alongside the oracle.

        INPUTS: tmp_path (pytest fixture), capsys (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path / "test")
        model = create_model(num_classes=3, pretrained=False)
        save_checkpoint(tmp_path / "m.pt", model, ["bicycle", "bus", "hydrant"])

        code = cli.main(["bench", "--data", str(tmp_path / "test"), "-n", "3",
                         "--solver", "oracle", "model", "--checkpoint", str(tmp_path / "m.pt"),
                         "--device", "cpu"])

        output = capsys.readouterr().out
        assert code == 0
        assert "bus: 2" in output
        assert any(line.startswith("model") for line in output.splitlines())

    def test_bench_results_match_seed(self):
        """
        FUNCTION NAME: test_bench_results_match_seed

        DESCRIPTION: The same seed gives the same results, so
        tables from different runs are comparable.

        INPUTS: None

        OUTPUTS: None
        """
        args = cli.build_parser().parse_args(["bench", "-n", "30", "--seed", "4"])
        assert cli.run_bench(args) == cli.run_bench(args)

    def test_bench_errors(self, tmp_path, capsys):
        """
        FUNCTION NAME: test_bench_errors

        DESCRIPTION: A missing checkpoint, a missing data
        folder, or an invalid p prints an error and returns 1;
        an unknown solver is an argparse error.

        INPUTS: tmp_path (pytest fixture), capsys (pytest fixture)

        OUTPUTS: None
        """
        assert cli.main(["bench", "--solver", "model",
                         "--checkpoint", str(tmp_path / "none.pt")]) == 1
        assert "python -m Library.train" in capsys.readouterr().err

        assert cli.main(["bench", "--data", str(tmp_path / "missing")]) == 1
        assert cli.main(["bench", "--solver", "noisy", "--p", "1.5"]) == 1
        with pytest.raises(SystemExit):
            cli.main(["bench", "--solver", "cnn"])

    def test_bench_warns_on_train_split(self, tmp_path, capsys):
        """
        FUNCTION NAME: test_bench_warns_on_train_split

        DESCRIPTION: Benchmarking a folder named train prints a
        warning.

        INPUTS: tmp_path (pytest fixture), capsys (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path / "train")
        assert cli.main(["bench", "--data", str(tmp_path / "train"), "-n", "2",
                         "--solver", "oracle"]) == 0
        assert "training split" in capsys.readouterr().err

    def test_play_launches_gui(self, monkeypatch):
        """
        FUNCTION NAME: test_play_launches_gui

        DESCRIPTION: play builds a seeded env and runs the GUI
        (a fake CaptchaApp stands in, so no window or tkinter
        is needed).

        INPUTS: monkeypatch (pytest fixture)

        OUTPUTS: None
        """
        launched = []

        class FakeApp:
            """
            CLASS NAME: FakeApp

            DESCRIPTION: Records the env instead of opening a window.
            """
            def __init__(self, env):
                """
                FUNCTION NAME: __init__

                DESCRIPTION: Stores the env.

                INPUTS: env (CaptchaEnv)

                OUTPUTS: None
                """
                self.env = env

            def run(self):
                """
                FUNCTION NAME: run

                DESCRIPTION: Records that the app was run.

                INPUTS: None

                OUTPUTS: None
                """
                launched.append(self.env)

        monkeypatch.setitem(sys.modules, "Library.gui", types.SimpleNamespace(CaptchaApp=FakeApp))

        assert cli.main(["play", "--seed", "7"]) == 0
        assert len(launched) == 1
        assert launched[0].seed == 7
