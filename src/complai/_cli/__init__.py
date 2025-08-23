import typer

from complai._cli.eval import eval_command
from complai._cli.eval_retry import eval_retry_command
from complai._cli.list import list_command


# Create a Typer instance
app = typer.Typer(rich_markup_mode="markdown")

# Add commands
app.command("eval")(eval_command)
app.command("eval-retry")(eval_retry_command)
app.command("list")(list_command)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
