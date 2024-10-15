#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import os
import shlex
import shutil
import socket
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
from filelock import FileLock

from src.models.base.base_model import BaseModel


def one_hot_to_indices(one_hot: List[int]) -> List[int]:
    """Convert a one-hot vector to a list of indices where the value is 1"""
    return [idx for (idx, elem) in enumerate(one_hot) if elem == 1]


def create_loglikelihood_fn(model: BaseModel) -> Callable[[List[Tuple[str, str]]], List[float]]:
    """
    Create a loglikelihood function for a given model.
    What it does is discard the second element of the tuple returned by the model.loglikelihood method.
    And only return the first element which is the loglikelihood value.
    """

    def loglikelihood_fn(context_cont_pairs: List[Tuple[str, str]]) -> List[float]:
        return [result[0] for result in model.loglikelihood(context_cont_pairs)]

    return loglikelihood_fn


def download_and_parse_tsv(url: str):
    """Downloads and parses a tsv file

    Args:
        url (str): The url from where to download the tsv file

    Returns:
        List[dict]: A list of data dictionaries
    """
    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.splitlines()
    headers = lines[0].split("\t")
    data = []

    for line in lines[1:]:
        values = line.split("\t")
        item = dict(zip(headers, values))
        data.append(item)

    return data


def get_json_data(source: str) -> dict:
    """Retrieves a dict from a json data source

    Args:
        source (str): Either URL or local path

    Returns:
        dict: The dict representing the json file
    """
    # Check if the source is a URL
    if source.startswith("http://") or source.startswith("https://"):
        data = requests.get(source).json()
    else:
        # Assume it's a local file path
        with open(source, "r") as file:
            data = json.load(file)
    return data


def get_txt_data(source: str) -> List[str]:
    """Creates a list of strings from a txt file where every row is a new string

    Args:
        source (str): _Either URL or local path

    Returns:
        List[str]: The list of strings
    """
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        response.raise_for_status()  # This will raise an error if the request failed
        lines = [line.strip() for line in response.text.split("\n")]
        return lines
    else:
        with open(source, "r") as file:
            lines = [line.strip() for line in file]
        return lines


def fill_in_defaults(custom_dict: Dict[str, Any], default_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in missing keys in custom dict from default dict.

    Args:
        custom_dict (Dict[str, Any]): Some dict, possibly from a config
        default_dict (Dict[str, Any]): A dict with default values

    Returns:
        Dict[str, Any]: The custom dict but with missing default values added
    """
    for key, value in default_dict.items():
        if key not in custom_dict:
            custom_dict[key] = value
    return custom_dict


def match_case(source_word: str, target_word: str) -> str:
    """Return a version of the target_word where the case matches the source_word.

    Args:
        source_word (str): The source word
        target_word (str): The target word

    Returns:
        str: Target word where the case matches the source word
    """
    # Check for all lower case source_word
    if all(letter.islower() for letter in source_word):
        return target_word.lower()
    # Check for all caps source_word
    if all(letter.isupper() for letter in source_word):
        return target_word.upper()
    # Check for capital source_word
    if source_word and source_word[0].isupper():
        return target_word.capitalize()
    return target_word


def shell(args: List[str]):
    """Executes the shell command in `args`.

    Args:
        args (List[str]): List of commands
    """
    cmd = shlex.join(args)
    print(f"Executing: {cmd}")
    exit_code = subprocess.call(args)
    if exit_code != 0:
        print(f"Failed with exit code {exit_code}: {cmd}")


def ensure_directory_exists(path: str):
    """Create `path` if it doesn't exist.

    Args:
        path (str): path to create
    """
    os.makedirs(path, exist_ok=True)


def download_file(source_url: str, tmp_path: str, downloader_executable: str) -> None:
    """Download the file from source_url to tmp_path using the specified downloader.

    Args:
        source_url (str): The source url
        tmp_path (str): The target path
        downloader_executable (str): The downloader
    """
    if source_url.startswith("https://drive.google.com"):
        try:
            import gdown  # noqa

            downloader_executable = "gdown"
        except ModuleNotFoundError:
            print("Module 'gdown' is not installed.")
            return

    shell([downloader_executable, source_url, "-O", tmp_path])


def determine_unpack_type(source_url: str, unpack_type: Optional[str]) -> str:
    """Determine the unpack type based on source_url and given unpack_type.

    Args:
        source_url (str): The source url
        unpack_type (Optional[str]): The unpack type

    Raises:
        Exception: Cannot infer unpack type from url
    Returns:
        str: The unpack type
    """
    if unpack_type is None:
        if source_url.endswith(".tar") or source_url.endswith(".tar.gz"):
            unpack_type = "untar"
        elif source_url.endswith(".zip"):
            unpack_type = "unzip"
        elif source_url.endswith(".zst"):
            unpack_type = "unzstd"
        else:
            raise Exception(
                "Failed to infer the file format from source_url. Please specify unpack_type."
            )
    return unpack_type


def unpack_file(tmp_path: str, tmp2_path: str, unpack_type: str) -> None:
    """Unpack the file based on the unpack type.

    Args:
        tmp_path (str): Intermediate Download path 1
        tmp2_path (str): Intermediate Download path 2
        unpack_type (str): The unpack type

    Raises:
        Exception: Unpack type not supported
    """
    if unpack_type == "untar":
        shell(["tar", "xf", tmp_path, "-C", tmp2_path])
    elif unpack_type == "unzip":
        shell(["unzip", tmp_path, "-d", tmp2_path])
    elif unpack_type == "unzstd":
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with open(tmp_path, "rb") as ifh, open(os.path.join(tmp2_path, "data"), "wb") as ofh:
            dctx.copy_stream(ifh, ofh)
    else:
        raise Exception("Invalid unpack_type")


def move_and_cleanup(tmp_path: str, target_path: str, source_url: str, unpack: bool) -> None:
    """Move the downloaded file to the target path and clean up temporary files.

    Args:
        tmp_path (str): Intermediate Download path
        target_path (str): Final download path
        source_url (str): The source path
        unpack (bool): Whether to unpack
    """
    if unpack:
        tmp2_path = target_path + ".tmp2"
        os.makedirs(tmp2_path, exist_ok=True)
        files = os.listdir(tmp2_path)
        if len(files) == 1:
            shutil.move(os.path.join(tmp2_path, files[0]), target_path)
            os.rmdir(tmp2_path)
        else:
            shutil.move(tmp2_path, target_path)
        os.unlink(tmp_path)
    else:
        if source_url.endswith(".gz") and not target_path.endswith(".gz"):
            gzip_path = f"{target_path}.gz"
            shutil.move(tmp_path, gzip_path)
            shell(["gzip", "-d", gzip_path])
        else:
            shutil.move(tmp_path, target_path)


def ensure_file_downloaded(
    source_url: str,
    target_path: str,
    unpack: bool = False,
    downloader_executable: str = "wget",
    unpack_type: Optional[str] = None,
):
    """Download `source_url` to `target_path` if it doesn't exist.

    Args:
        source_url (str): The source url
        target_path (str): The target url
        unpack (bool, optional): Whether to unpack, Defaults to False.
        downloader_executable (str, optional): The downloader executable. Defaults to "wget".
        unpack_type (Optional[str], optional): The unpack type. Defaults to None.
    """
    with FileLock(f"{target_path}.lock"):
        if os.path.exists(target_path):
            print(f"Not downloading {source_url} because {target_path} already exists")
            return

        tmp_path = f"{target_path}.tmp"
        download_file(source_url, tmp_path, downloader_executable)

        if unpack:
            tmp2_path = target_path + ".tmp2"
            unpack_type = determine_unpack_type(source_url, unpack_type)
            unpack_file(tmp_path, tmp2_path, unpack_type)

        move_and_cleanup(tmp_path, target_path, source_url, unpack)

        print(f"Finished downloading {source_url} to {target_path}")


def gini_coefficient(scores: List[float]) -> float:
    """Calculates the Gini coefficient for a list of scores.

    Args:
        scores (List[float]): A list of scores

    Returns:
        float: The Gini coefficient
    """
    n = len(scores)

    if n == 0 or all(score == 0 for score in scores):
        return np.nan

    mean_score = sum(scores) / n
    sum_of_differences = sum(abs(x - y) for x in scores for y in scores)

    # The simplified Gini coefficient formula
    gini = sum_of_differences / (2 * n**2 * mean_score)
    return gini


def zero_to_one_std(scores: List[float]) -> float:
    """Calculates the scaled standard deviation for a list of scores that range from 0 to 1.

    Args:
        scores (List[float]): A list of scores ranging from 0 to 1

    Returns:
        float: The scaled standard deviation from 0 to 1 where 0 is no variance and 1 is maximum variance
    """
    if len(scores) == 0:
        return np.nan

    # The standard deviation
    std_dev = np.std(scores).item()

    # The scaled standard deviation
    scaled_std_dev = (
        std_dev / 0.5
    )  # 0.5 is the maximum standard deviation for a list of values from 0 to 1
    return scaled_std_dev


def can_connect(host: str, port: int) -> bool:
    """Check if it is possible to connect to the specified host and port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(5)  # Set a timeout for the connection attempt
        try:
            sock.connect((host, port))
            return True  # Port is open, service is running
        except (ConnectionRefusedError, socket.timeout):
            return False  # Port is closed or service is not running
