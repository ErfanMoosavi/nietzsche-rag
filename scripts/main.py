from .etl import run_etl
from .save_pretrained_model import save_model


def main() -> None:
    save_model()
    run_etl()


if __name__ == "__main__":
    main()
