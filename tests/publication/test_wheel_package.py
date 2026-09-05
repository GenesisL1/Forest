from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class WheelPackageTests(unittest.TestCase):
    def test_wheel_contains_mint_module_and_console_scripts(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            source = temp / "source"
            wheelhouse = temp / "wheelhouse"
            source.mkdir()
            wheelhouse.mkdir()

            for name in (
                "pyproject.toml",
                "README.md",
                "train_gl1f.py",
                "local_trainer_server.py",
                "gl1f_validate.py",
                "mint_model.py",
                "mint_workflow.py",
            ):
                shutil.copy2(repository / name, source / name)

            environment = {
                **os.environ,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
                "SOURCE_DATE_EPOCH": "1704067200",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheelhouse),
                    str(source),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            wheels = list(wheelhouse.glob("gl1f_forest-0.2.3-*.whl"))
            self.assertEqual(len(wheels), 1, result.stdout)

            with zipfile.ZipFile(wheels[0]) as archive:
                names = set(archive.namelist())
                self.assertIn("mint_model.py", names)
                self.assertIn("mint_workflow.py", names)
                self.assertIn("gl1f_validate.py", names)
                entry_name = next(
                    name for name in names if name.endswith(".dist-info/entry_points.txt")
                )
                parser = configparser.ConfigParser()
                parser.read_string(archive.read(entry_name).decode("utf-8"))
                self.assertEqual(parser["console_scripts"]["gl1f-train"], "train_gl1f:main")
                self.assertEqual(parser["console_scripts"]["gl1f-mint"], "mint_model:main")


if __name__ == "__main__":
    unittest.main()
