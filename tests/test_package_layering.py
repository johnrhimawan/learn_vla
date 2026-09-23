"""Enforce the package layering so imports cannot start pointing backwards.

The layering is the reason the physics layer can be trusted as a reference:
it knows nothing about the robot, the scene, or the planner, so calibrating
against it cannot become circular.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "tennis_vla"

# Each package may import only from itself and the packages listed here.
ALLOWED_DEPENDENCIES: dict[str, set[str]] = {
    "physics": set(),
    "robot": set(),
    "reporting": set(),
    "environment": {"physics", "robot"},
    "planning": {"physics", "robot", "environment"},
    "control": {"physics", "robot", "environment", "planning"},
    "perception": {"physics", "robot", "environment", "reporting"},
}


def _imported_packages(path: Path) -> set[str]:
    """Return the sibling packages a module imports from."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    packages = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module:
            packages.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tennis_vla."):
                    packages.add(alias.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module and node.module.startswith("tennis_vla."):
                packages.add(node.module.split(".")[1])
    return packages


class PackageLayeringTests(unittest.TestCase):
    def test_every_package_is_declared(self) -> None:
        found = {
            path.name
            for path in PACKAGE_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }
        self.assertEqual(found, set(ALLOWED_DEPENDENCIES))

    def test_imports_only_point_down_the_layers(self) -> None:
        violations = []
        for package, allowed in ALLOWED_DEPENDENCIES.items():
            for path in sorted((PACKAGE_ROOT / package).rglob("*.py")):
                for imported in _imported_packages(path) - allowed - {package}:
                    violations.append(
                        f"{path.relative_to(PACKAGE_ROOT.parent)} imports "
                        f"'{imported}', which {package} may not depend on"
                    )
        self.assertEqual(violations, [])

    def test_no_module_sits_outside_a_package(self) -> None:
        stray = [
            path.name
            for path in PACKAGE_ROOT.glob("*.py")
            if path.name != "__init__.py"
        ]
        self.assertEqual(stray, [])

    def test_subpackages_reexport_what_they_declare(self) -> None:
        import importlib

        for package in ALLOWED_DEPENDENCIES:
            module = importlib.import_module(f"tennis_vla.{package}")
            missing = [
                name for name in module.__all__ if not hasattr(module, name)
            ]
            self.assertEqual(missing, [], f"tennis_vla.{package}")


if __name__ == "__main__":
    unittest.main()
