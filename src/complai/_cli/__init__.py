import typer
from inspect_ai._util.dotenv import init_dotenv

from complai._cli.eval import eval_command
from complai._cli.list import list_command
from complai._cli.samples import samples_command


app = typer.Typer(rich_markup_mode="markdown")

app.command("eval")(eval_command)
app.command("list")(list_command)
app.command("samples")(samples_command)


def main() -> None:
    init_dotenv()
    app()


if __name__ == "__main__":
    main()
