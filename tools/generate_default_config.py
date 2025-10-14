"""This script generates a YAML file listing the default values for each task parameter."""

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import list_tasks
from inspect_ai import TaskInfo


def format_type_annotation(annotation_node: ast.expr | None) -> str:
    """Format an AST type annotation into a readable string."""
    if annotation_node is None:
        return "any"

    return ast.unparse(annotation_node)


def eval_default_value(default_node: ast.expr) -> Any:
    """Safely evaluate a default value from an AST node."""
    try:
        # Use ast.literal_eval for safe evaluation of literals
        return ast.literal_eval(default_node)
    except (ValueError, SyntaxError):
        # If it's not a literal, return the unparsed string representation
        return ast.unparse(default_node)


def find_task_func_node(tree: ast.Module, task_name: str) -> ast.FunctionDef | None:
    """Find a function with the @task decorator in the AST."""
    task_func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                # Check if this is a @task decorator
                is_task_decorator = False
                decorator_call = None
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "task"
                ):
                    is_task_decorator = True
                    decorator_call = decorator
                elif isinstance(decorator, ast.Name) and decorator.id == "task":
                    is_task_decorator = True

                if is_task_decorator:
                    # It's a task function, now check if it's the one we're looking for
                    # Default task name is the function name
                    effective_task_name = node.name
                    if decorator_call:
                        for kw in decorator_call.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                effective_task_name = kw.value.value  # type: ignore
                                break

                    if effective_task_name == task_name:
                        task_func_node = node
                        break  # decorator loop
        if task_func_node:
            break  # functiondef loop

    return task_func_node


def extract_task_defaults(task_info: TaskInfo) -> dict[str, dict[str, Any]]:
    """Extract parameters and their defaults from a task function.

    Returns:
        Tuple of (defaults_dict, task_description) where task_description is the first line
        of the docstring or None.
    """
    task_module_path = Path(task_info.file)
    task_name = task_info.name

    # Read and parse the file
    try:
        source_code = task_module_path.read_text()
        tree = ast.parse(source_code)
    except Exception:
        print(f"[WARNING] Error parsing task {task_name} from {task_module_path}")
        return {}

    # Find the task function definition
    task_func_node = find_task_func_node(tree, task_name)

    if task_func_node is None:
        return {}

    # Extract default values
    defaults_dict: dict[str, dict[str, Any]] = defaultdict(dict)
    params = task_func_node.args.args
    defaults = task_func_node.args.defaults
    # Defaults are aligned to the right (args with defaults are last)
    for param, default in zip(params[-len(defaults) :], defaults):
        param_name = param.arg
        defaults_dict[param_name] = eval_default_value(default)

    return defaults_dict


def generate_default_config_yaml(task_dir: Path) -> str:
    """Generate the default configuration YAML content."""
    # Get all tasks
    tasks = list_tasks(absolute=True, root_dir=task_dir)
    tasks.sort(key=lambda t: t.name)

    # Build YAML content
    lines = [
        "# Default Task Configuration File",
        "# ================================",
        "# This file shows all COMPL-AI tasks with configurable arguments and their default values.",
        "# To customize the configuration you can copy this file and modify it.",
        "# Usage: complai eval <model> --task-config default_config.yaml",
        "#",
        "# This file was automatically generated from task code.",
        "# To regenerate: uv run --with inspect-ai pyyaml --no-project tools/generate_default_config.py",
        "",
        "",
        "",
    ]

    for task_info in tasks:
        # Get task parameters and description
        defaults = extract_task_defaults(task_info)

        if defaults:
            lines.append(f"{task_info.name}:")
            for param_name, default_val in defaults.items():
                # Use yaml.dump to format the value correctly, then format for inline
                # We dump a dict to ensure the key is present, then strip the key
                # e.g., {'my_key': None} -> 'my_key: null\n' -> 'null'
                dumped = yaml.dump({param_name: default_val}, default_flow_style=True)
                yaml_val = dumped.split(":", 1)[1].strip().replace("\n", "").rstrip("}")
                lines.append(f"  {param_name}: {yaml_val}")

            lines.append("")

    return "\n".join(lines)


def main(
    task_dir: Path = Path(__file__).parent.parent / "src" / "complai" / "tasks",
    output_path: Path = Path(__file__).parent.parent / "config" / "default_config.yaml",
) -> int:
    yaml_content = generate_default_config_yaml(task_dir)

    # Verify it's valid YAML
    try:
        yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        print(f"✗ Error: Generated YAML is invalid: {e}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content)
    print(f"✓ Generated {output_path}")
    print(f"  Total lines: {len(yaml_content.splitlines())}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
