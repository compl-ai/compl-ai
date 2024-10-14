import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path

from src.models.base.model_factory import get_model_from_config
from src.registry import registry
from src.results.base_connector import BenchmarkInfo
from src.results.file_connector import FileConnector
from src.runner import Runner
from src.utils.initialization import seed_everything
from src.utils.loading import parse_config, patch_config, read_config_from_yaml
from src.utils.log_manager import log_manager

logging.basicConfig(level=logging.DEBUG)


def my_app(cfg: dict, results_folder=Path("runs")) -> None:
    """
    Run the benchmark specified by the configuration.

    Args:
        config (dict): The configuration for the benchmark run.
        run_name (str, optional): The name of the run. Defaults to a random name.
        results_folder (str, optional): The folder to store the results. Defaults to "runs".
        category (str, optional): The category name of the benchmark. Defaults to None.
        benchmark (str, optional): The name of the benchmark to run. Defaults to None.

    Returns:
        None
    """
    # Set up logging
    log_folder = Path("logs")
    log_manager.set_logger_folder(log_folder)
    logger = log_manager.get_new_logger("main")

    # parsing config
    cfg_obj = parse_config(cfg)
    print("Patched config: ")
    print(cfg_obj)

    # Set seed
    seed_everything(cfg_obj.seed)

    # Initialize model
    model = get_model_from_config(cfg_obj.model)

    benchmark_name = cfg_obj.benchmark_configs[0].name
    benchmark_type = cfg_obj.benchmark_configs[0].type
    category = registry.get("benchmark").get_category(benchmark_type)

    # Name and not type since name is unique per benchmark
    benchmark_info = BenchmarkInfo(benchmark_type=benchmark_name, category=category)

    # Setup database handler
    result_handler = FileConnector(benchmark_info, run_folder=results_folder, create_run=True)
    result_handler.store_config(cfg_obj)

    # Initializer runner
    runner = Runner(model, cfg_obj, result_handler)

    # Log the start time
    start_time = datetime.now()
    date_format = "%Y-%m-%d_%H:%M:%S"
    datetime_string = start_time.strftime(date_format)
    logger.info(f"Run started at {datetime_string}")

    # Run
    try:
        runner.run()

    except Exception as e:
        traceback_str = traceback.format_exc()
        logging.error("An error occurred during the benchmark:\n%s", traceback_str)
        raise e

    finally:
        # Log the end time
        end_time = datetime.now()
        datetime_string = end_time.strftime(date_format)
        logger.info(f"Run ended at {datetime_string}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EU AI Checker tool")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument(
        "--profiling", action="store_true", default=False, help="Whether or not to run profiling"
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        help="Name of the run folder where results from all the benchmarks are stored",
    )
    parser.add_argument("--model", type=str, help="Name of model to use")
    parser.add_argument(
        "--model_config", type=str, help="Config file form model, applied before --model"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size to use")
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        default=False,
        help="When in debug mode, checker is only run on subset of data",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        help="If in debug mode, which subset size to use",
    )
    parser.add_argument(
        "--cpu_mode",
        action="store_true",
        default=False,
        help="When in cpu mode, benchmark are run such that they can be run on cpu for testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default=None,
        help="Which device to use, overwrites the device from model configs",
    )
    parser.add_argument("--answers_file", default=None, type=str, help="Path to the answers file")

    args = parser.parse_args()

    # Access the parsed file path argument
    config_path = args.config_path
    results_folder = args.results_folder
    if not results_folder:
        results_folder = Path("runs")
    else:
        results_folder = Path(results_folder)
    model = args.model
    model_config = args.model_config
    batch_size = args.batch_size
    debug_mode = args.debug_mode
    answers_file = args.answers_file
    cpu_mode = args.cpu_mode
    subset_size = args.subset_size
    device = args.device

    run_name = config_path.split("/")[2].split(".")[0]
    config_dict = read_config_from_yaml(config_path)
    config_dict = patch_config(
        config_dict,
        model,
        model_config,
        batch_size,
        debug_mode,
        answers_file,
        cpu_mode,
        subset_size,
        device,
    )
    logging.debug("Patched config dict: ", config_dict)

    my_app(config_dict, results_folder=results_folder)
