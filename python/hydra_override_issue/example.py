# For `uv run`.
# /// script
# dependencies = [
#   "hydra-core==1.3.2",
# ]
# ///

from pathlib import Path
import hydra


@hydra.main(version_base=None, config_path=str(Path(__file__).parent))
def main(cfg):
    print(cfg)


if __name__ == "__main__":
    main()
