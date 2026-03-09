import argparse
import os
import shutil
import signal
import stat
import subprocess
import sys
from time import sleep

try:
    # Cross platform and python native
    import psutil
    PSUTIL = True
except ModuleNotFoundError:
    PSUTIL = False

def clean_dir(path):
    def onerror(func, path, exc_info):
        if not os.access(path, os.W_OK):
            # Is the error an access error ?
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise
    retry_timeout = 2
    for retry_count in range(5):
        try:
            shutil.rmtree(path, onerror=onerror)
            os.mkdir(path)
            return
        except PermissionError:
            # For when the OS hasn't marked all of the files no longer in use.
            sleep(retry_timeout)
            retry_timeout = retry_timeout ** 2
    print(f"ERROR: Failed to clean {path} after timeout")
    sys.exit(1)


def taskkill(pid, tree=False, force=False):
    if PSUTIL:
        cmd_proc = psutil.Process(pid)
        if tree:
            children = cmd_proc.children(recursive=True)
        else:
            children = []
        cmd_proc.send_signal(signal.SIGTERM)
        print(f"\tkilled {cmd_proc.name()}", flush=True)
        sys.stdout.flush()
        for child in children:
            try:
                child.send_signal(signal.SIGTERM)
                print(f"\tkilled {child.name()}", flush=True)
            except psutil.NoSuchProcess:
                pass
    else:
        taskkill_call = ["taskkill"]
        if tree:
            taskkill_call.append("/T")
        if force:
            taskkill_call.append("/F")
        taskkill_call += ["/PID", str(pid)]
        return subprocess.run(taskkill_call)


def run_with_timeout_retry(args, retries, timeout, workspace):
    for attempt in range(retries):
        with subprocess.Popen(args) as proc:
            try:
                ret = proc.wait(timeout)
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, args)
                else:
                    return ret
            except subprocess.TimeoutExpired as e:
                print(f"Attempt {attempt+1} timed out after {timeout} seconds",
                      flush=True)
                taskkill(proc.pid, tree=True, force=True)
                if attempt + 1 >= retries:
                    print(f"ERROR: Exhausted all retries. Exiting", flush=True)
                    sys.exit(1)
                if workspace is not None:
                    clean_dir(workspace)
                sleep(5)
                print(f"Retrying command:\n{' '.join(args)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timeout", help="Timeout in seconds", type=int)
    parser.add_argument("-r", "--retries", help="Number of attempts", type=int)
    parser.add_argument("-w",
                        "--workspace",
                        help="Workspace to be cleared",
                        type=str)
    args, call_args = parser.parse_known_args()
    print(
        f"Attempting to run the following command with {args.timeout} second timeout and {args.retries} retries:\n{' '.join(call_args)}",
        flush=True)
    run_with_timeout_retry(call_args, args.retries, args.timeout,
                           args.workspace)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
