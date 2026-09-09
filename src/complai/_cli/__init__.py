import typer
from inspect_ai._util.dotenv import init_dotenv

from complai._cli.predict import predict_command
from complai._cli.eval import eval_command
from complai._cli.list import list_command
from complai._cli.samples import samples_command


app = typer.Typer(rich_markup_mode="markdown")

app.command("eval")(eval_command)
app.command("list")(list_command)
app.command("samples")(samples_command)
app.command("predict")(predict_command)



import sys
from pathlib import Path

# Conditionally load maintainer tools if we are in the source repository
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if (_repo_root / "tools" / "minify").exists():
    try:
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from tools.minify.minify import app as minify_app
        app.add_typer(minify_app, name="minify", hidden=True, help="Maintainer tools for subset generation.")
    except ImportError:
        pass

def main() -> None:
    init_dotenv()
    app()


if __name__ == "__main__":
    main()
