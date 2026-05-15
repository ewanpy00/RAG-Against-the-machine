import fire

from student.cli import CLI


def main() -> None:
    """Entry point for the student CLI."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
