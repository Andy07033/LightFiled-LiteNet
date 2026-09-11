"""Scratch batch32 ~2M 3x3 RGB training: two-layer stems, step-2 inference, angular mixing."""
import importlib.util


if __name__ == "__main__":
    packages = {"numpy": "NumPy", "torch": "PyTorch", "yaml": "PyYAML", "PIL": "Pillow"}
    missing = [label for name, label in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Missing Python libraries: " + ", ".join(missing)
            + ". Use the Python environment containing these libraries. "
            "No packages or services have been installed by this script. See README.md."
        )
    from lfnav.train import main
    main(default_config="configs/train_stem2_step2_2m_batch32_angularmix_3090.yaml")
