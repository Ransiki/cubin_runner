from datetime import datetime
import os
import subprocess
import json

def getGitCommitDate(source_dir, commit):
    try:
        output = subprocess.check_output(
            ["git", "show", "-s", "--format=%at", commit],
            stderr=subprocess.STDOUT,
            cwd=source_dir
        )
    except subprocess.CalledProcessError as exc:
        print("Failed to read commit date", exc.returncode, exc.output)
        raise exc
    
    commit_epoch = int(output.decode('ascii').strip())

    return datetime.fromtimestamp(commit_epoch)

def getCudaArtifact(cuda, env=os.environ):
    # Case when running inside container: env (CUDA_ARTIFACT) is set inside container
    if "CUDA_ARTIFACT" in env:
        return env["CUDA_ARTIFACT"]
    # Case when running outside container (e.g. lsf)
    else:
        artifact_url = ""
        if "gpgpu_internal" in cuda.lower():
            revision = getCudaRevision()
            if revision:
                return f"https://urm.nvidia.com/artifactory/sw-fastkernels-generic/cuda/gpgpu/x86_64/linux/generic/release-internal/cuda-gpgpu-{revision}.tgz"

        return artifact_url

def getCudaRevision(env=os.environ):
    revision = ""

    if "CUDA_REVISION" in env:
        revision = env["CUDA_REVISION"]
        print(f"Found CUDA_REVISION = {revision}")

    return revision

def getGitInfo():
    with open("scmProperties.json") as f:
        scm_properties = json.load(f)
    
    checkout = scm_properties["checkout"]
    branch = checkout["GIT_BRANCH"]

    if branch.startswith("origin/"):
        branch = branch[len("origin/"):]
    
    target_branch = scm_properties["branchName"]
    target_branch = target_branch.replace("release_", "release/")

    commit_id = checkout["GIT_COMMIT"]
    commit_date = getGitCommitDate(checkout["GIT_CHECKOUT_DIR"], commit_id)

    return {
        "branch": branch,
        "target_branch": target_branch,
        "commit_id": commit_id,
        "commit_date": commit_date
    }
