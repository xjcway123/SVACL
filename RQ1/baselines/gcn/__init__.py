from pathlib import Path
import inspect
from datetime import datetime
import subprocess
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

def project_dir() -> Path:
    """Get project path."""
    return Path(__file__).parent.parent
def graphdata_dir()->Path:
    return Path(__file__).parent.parent / "graphdata"


def get_dir(path) -> Path:
    """Get path, if exists. If not, create it."""
    Path(path).mkdir(exist_ok=True, parents=True)
    return path

def data_dir()->Path:
    return Path(__file__).parent.parent / "Data"

def function_Le_dir()->Path:
    return Path(__file__).parent / "function_level_Le"

def Deepcva_dir()->Path:
    return Path(__file__).parent / "Deepcva"
def codebert_dir()->Path:
    return Path(__file__).parent / "codebert"

def multi_codebert_dir()->Path:
    return Path(__file__).parent / "multi_codebert"

def embedding_graph_dir()->Path:
    return Path(__file__).parent / "embedding_graph_model"

def multi_embedding_graph_dir()->Path:
    return Path(__file__).parent / "multi_embedding_graph_model"
def script_dir()->Path:
    return Path(__file__).parent.parent / "script"
def multi_task_dir()->Path:
    return Path(__file__).parent / "multi_task"
def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\t\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )

def subprocess_cmd(command: str, verbose: int = 0, force_shell: bool = False):
    """Run command line process.

    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    >>> a
    >>> b
    """
    # singularity = os.getenv("SINGULARITY")
    # if singularity != "true" and not force_shell:
    #     command = f"singularity exec {project_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    return output

def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return x

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1)
    """
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    processed = []
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
    return processed
def gitsha():
    """Get current git commit sha for reproducibility."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )


def gitmessage():
    """Get current git commit sha for reproducibility."""
    m = subprocess.check_output(["git", "log", "-1", "--format=%s"]).strip().decode()
    return "_".join(m.lower().split())
def get_run_id(args=None):
    """Generate run ID."""
    if not args:
        ID = datetime.now().strftime("%Y%m%d%H%M_{}".format(gitsha()))
        return ID + "_" + gitmessage()
    ID = datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gitsha(), "_".join([f"{v}" for _, v in vars(args).items()])
        )
    )
    return ID