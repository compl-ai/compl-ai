import typer
from inspect_ai._util.dotenv import init_dotenv

from complai._cli.core import fit_command
from complai._cli.core import predict_command
from complai._cli.core import preprocess_command
from complai._cli.eval import eval_command
from complai._cli.list import list_command
from complai._cli.samples import samples_command


app = typer.Typer(rich_markup_mode="markdown")
core_app = typer.Typer(help="Build and use reduced evaluation sets.")

app.command("eval")(eval_command)
app.command("list")(list_command)
app.command("samples")(samples_command)
app.add_typer(core_app, name="core")
core_app.command("fit")(fit_command)
core_app.command("preprocess")(preprocess_command)
core_app.command("predict")(predict_command)


def main() -> None:
    init_dotenv()
    app()


if __name__ == "__main__":
    main()
