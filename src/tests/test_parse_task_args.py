import datetime
import tempfile
from pathlib import Path

import pytest
import typer

from complai._cli.utils import parse_task_args


class TestParseTaskArgs:
    """Test the parse_task_args function."""

    def test_empty_args(self) -> None:
        """Test with no CLI args and no config file."""
        result = parse_task_args([], None)
        assert result == {}

    def test_single_task_arg(self) -> None:
        """Test parsing a single task-specific argument."""
        result = parse_task_args(["bbq:shuffle=true"], None)
        assert result == {"bbq": {"shuffle": True}}

    def test_multiple_task_args_same_task(self) -> None:
        """Test multiple args for the same task."""
        result = parse_task_args(["bbq:shuffle=true", "bbq:subset=Age"], None)
        assert result == {"bbq": {"shuffle": True, "subset": "Age"}}

    def test_multiple_task_args_different_tasks(self) -> None:
        """Test args for multiple different tasks."""
        result = parse_task_args(
            ["bbq:shuffle=true", "bigbench_calibration:bigbench_task=emoji_movie"], None
        )
        assert result == {
            "bbq": {"shuffle": True},
            "bigbench_calibration": {"bigbench_task": "emoji_movie"},
        }

    def test_string_value(self) -> None:
        """Test parsing string values."""
        result = parse_task_args(["task1:name=test"], None)
        assert result == {"task1": {"name": "test"}}

    def test_boolean_value_true(self) -> None:
        """Test parsing boolean true values."""
        result = parse_task_args(["task1:enabled=true"], None)
        assert result == {"task1": {"enabled": True}}

    def test_boolean_value_false(self) -> None:
        """Test parsing boolean false values."""
        result = parse_task_args(["task1:enabled=false"], None)
        assert result == {"task1": {"enabled": False}}

    def test_float_value(self) -> None:
        """Test parsing float values."""
        result = parse_task_args(["task1:threshold=0.75"], None)
        assert result == {"task1": {"threshold": 0.75}}

    def test_integer_value(self) -> None:
        """Test parsing integer values."""
        result = parse_task_args(["task1:count=42"], None)
        assert result == {"task1": {"count": 42}}

    def test_comma_separated_value(self) -> None:
        """Test parsing comma-separated values (becomes list)."""
        result = parse_task_args(["task1:items=a,b,c"], None)
        assert result == {"task1": {"items": ["a", "b", "c"]}}

    def test_list_value(self) -> None:
        """Test parsing list values."""
        result = parse_task_args(["task1:items=[a,b,c]"], None)
        assert result == {"task1": {"items": ["a", "b", "c"]}}

    def test_value_with_equals_sign(self) -> None:
        """Test parsing values that contain equals signs."""
        result = parse_task_args(["task1:key=string=with=equals=sign"], None)
        assert result == {"task1": {"key": "string=with=equals=sign"}}

    def test_missing_colon_raises_error(self) -> None:
        """Test that missing colon raises BadParameter error."""
        with pytest.raises(
            typer.BadParameter, match="must be in format 'task_name:key=value'"
        ):
            parse_task_args(["arg=value"], None)

    def test_missing_equals_raises_error(self) -> None:
        """Test that missing equals sign raises BadParameter error."""
        with pytest.raises(
            typer.BadParameter, match="must be in format 'task_name:key=value'"
        ):
            parse_task_args(["task:arg"], None)

    def test_invalid_yaml_raises_error(self) -> None:
        """Test that invalid YAML in value raises BadParameter error."""
        with pytest.raises(typer.BadParameter, match="Could not parse value"):
            parse_task_args(["task:arg=[unclosed"], None)

    def test_config_file_yaml(self) -> None:
        """Test loading task args from a YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
bbq:
  shuffle: true
  subset: Age
bigbench_calibration:
  bigbench_task: emoji_movie
""")
            f.flush()

            try:
                result = parse_task_args([], f.name)
                assert result == {
                    "bbq": {"shuffle": True, "subset": "Age"},
                    "bigbench_calibration": {"bigbench_task": "emoji_movie"},
                }
            finally:
                Path(f.name).unlink()

    def test_config_file_json(self) -> None:
        """Test loading task args from a JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("""
{
  "bbq": {
    "shuffle": true
  },
  "bigbench_calibration": {
    "bigbench_task": "emoji_movie"
  }
}
""")
            f.flush()

            try:
                result = parse_task_args([], f.name)
                assert result == {
                    "bbq": {"shuffle": True},
                    "bigbench_calibration": {"bigbench_task": "emoji_movie"},
                }
            finally:
                Path(f.name).unlink()

    def test_cli_overrides_config(self) -> None:
        """Test that CLI args override config file values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
bbq:
  shuffle: true
  subset: Age
""")
            f.flush()

            try:
                result = parse_task_args(["bbq:shuffle=false"], f.name)
                assert result == {"bbq": {"shuffle": False, "subset": "Age"}}
            finally:
                Path(f.name).unlink()

    def test_cli_adds_to_config(self) -> None:
        """Test that CLI args can add new tasks not in config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
bbq:
  shuffle: true
""")
            f.flush()

            try:
                result = parse_task_args(["bbq:subset=Age"], f.name)
                assert result == {"bbq": {"shuffle": True, "subset": "Age"}}
            finally:
                Path(f.name).unlink()

    def test_dict_value_in_yaml_syntax(self) -> None:
        """Test parsing dict values using YAML syntax."""
        result = parse_task_args(["task1:config={key: value}"], None)
        assert result == {"task1": {"config": {"key": "value"}}}

    def test_empty_string_value(self) -> None:
        """Test parsing empty string values."""
        result = parse_task_args(["task1:key=''"], None)
        # Empty string after = should parse as ""
        assert "task1" in result
        assert "key" in result["task1"]
        assert result["task1"]["key"] == ""

    def test_none_value(self) -> None:
        """Test parsing empty string values."""
        result = parse_task_args(["task1:key="], None)
        # No value after = should parse as None
        assert "task1" in result
        assert "key" in result["task1"]
        assert result["task1"]["key"] is None

    def test_task_name_with_underscores(self) -> None:
        """Test task names with underscores."""
        result = parse_task_args(["my_task_name:arg=value"], None)
        assert result == {"my_task_name": {"arg": "value"}}

    def test_multiple_colons_in_value(self) -> None:
        """Test values containing multiple colons."""
        result = parse_task_args(["task1:url=http://example.com:8080"], None)
        assert result == {"task1": {"url": "http://example.com:8080"}}

    def test_special_characters_in_value(self) -> None:
        """Test values with special characters."""
        result = parse_task_args(["task1:pattern=.*[a-z]+"], None)
        assert result == {"task1": {"pattern": ".*[a-z]+"}}

    def test_whitespace_in_value(self) -> None:
        """Test values with whitespace (quoted strings)."""
        result = parse_task_args(["task1:message='hello world'"], None)
        assert result == {"task1": {"message": "hello world"}}

    def test_numeric_string_value(self) -> None:
        """Test numeric strings are parsed as numbers, not strings."""
        result = parse_task_args(["task1:port=8080"], None)
        assert result == {"task1": {"port": 8080}}
        assert isinstance(result["task1"]["port"], int)

    def test_date_value(self) -> None:
        """Test date values."""
        result = parse_task_args(["task1:date=2021-01-01"], None)
        assert result == {"task1": {"date": datetime.date(2021, 1, 1)}}
        assert isinstance(result["task1"]["date"], datetime.date)

    def test_mixed_cli_and_config_args(self) -> None:
        """Test CLI and config args together."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
task1:
  arg1: value1
  arg2: value2
task2:
  arg3: value3
""")
            f.flush()

            try:
                result = parse_task_args(
                    ["task1:arg2=overridden", "task1:arg4=new_value", "task3:arg5=100"],
                    f.name,
                )
                assert result == {
                    "task1": {
                        "arg1": "value1",
                        "arg2": "overridden",
                        "arg4": "new_value",
                    },
                    "task2": {"arg3": "value3"},
                    "task3": {"arg5": 100},
                }
            finally:
                Path(f.name).unlink()
