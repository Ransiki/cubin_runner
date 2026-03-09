import logging
import os
import mimetypes
from swiftclient.service import SwiftService, SwiftError, SwiftUploadObject
from swiftclient.exceptions import ClientException

logger = logging.getLogger(__name__)

mimetypes.add_type("text/plain", "log")
mimetypes.add_type("text/plain", "out")

SWIFTSTACK_ENDPOINT = "https://swiftstack-maglev.ngc.nvidia.com"
SWIFTSTACK_AUTH_VERSION = "1.0"
SWIFTSTACK_AUTH = "https://swiftstack-maglev.ngc.nvidia.com/auth/v1.0"


def construct_url(path, container):
    return f"{SWIFTSTACK_ENDPOINT}/v1/AUTH_cnexus/{container}/{path}"


def test_access(timeout=5, max_attempts=1):
    try:
        with SwiftService(
            {
                "auth_version": SWIFTSTACK_AUTH_VERSION,
                "auth": SWIFTSTACK_AUTH,
                "user": os.environ.get("ST_USER"),
                "key": os.environ.get("ST_KEY"),
                "os_storage_url": "https://swiftstack-maglev.ngc.nvidia.com/v1/AUTH_cnexus",
            }
        ) as swift:
            r = swift.stat(container="pytest")
            if r["success"]:
                return True
            else:
                return False
    except (SwiftError, ClientException) as e:
        logging.error(e.value)
        return False


def upload_files(files, container):
    """
    Validate the given container and upload all specified files

    :param files: dictionary containing the src and dest of each upload
    :param container: the container to upload to
    :return: An array of upload statuses
    """

    try:
        with SwiftService(
            {
                "auth_version": SWIFTSTACK_AUTH_VERSION,
                "auth": SWIFTSTACK_AUTH,
                "user": os.environ.get("ST_USER"),
                "key": os.environ.get("ST_KEY"),
            }
        ) as swift:
            # Create upload object for each file
            objs = [
                SwiftUploadObject(file["src"], object_name=file["dest"])
                for file in files
            ]

            status = []
            # Upload each file
            for r in swift.upload(container, objs):
                if r["success"]:
                    status.append(True)
                else:
                    error = r["error"]
                    if r["action"] == "create_container":
                        logging.error(f"Failed to create container {container}")
                        logging.error(error)
                    elif r["action"] == "upload_object":
                        logging.error(
                            f"Failed to upload object {r['object']} to container {container}"
                        )
                        logging.error(error)
                    else:
                        logging.error(error)
    except (SwiftError, ClientException) as e:
        logging.error(e.value)
        status = [False] * len(files)

    # Account for failures not caught by except
    if len(status) < len(files):
        status.extend([False] * (len(files) - len(status)))

    return status


def _verify_file(src, container):
    """
    Verify that a file exists

    :param src: path to the file in remote storage
    :param container: the container to check
    :return: True if the file exists, false otherwise
    """

    try:
        with SwiftService(
            {
                "auth_version": SWIFTSTACK_AUTH_VERSION,
                "auth": SWIFTSTACK_AUTH,
                "user": os.environ.get("ST_USER"),
                "key": os.environ.get("ST_KEY"),
            }
        ) as swift:
            for r in swift.stat(container=container, objects=[src]):
                if r["success"]:
                    return True
                else:
                    return False
    except (SwiftError, ClientException) as e:
        logging.error(e.value)
    return False


def download_file(src, dest, container):
    """
    Download a single file from the specified container

    :param src: path to the file in remote storage
    :param dest: filepath to download to
    :param container: the container to download from
    :return: True on a sucessful download, False otherwise
    """

    try:
        with SwiftService(
            {
                "auth_version": SWIFTSTACK_AUTH_VERSION,
                "auth": SWIFTSTACK_AUTH,
                "user": os.environ.get("ST_USER"),
                "key": os.environ.get("ST_KEY"),
            }
        ) as swift:
            for r in swift.download(container=container, objects=[src]):
                if r["success"]:
                    return True
                else:
                    error = r["error"]
                    logging.error(f"File download failed: {src}")
                    logging.error(error)
                    return False
    except (SwiftError, ClientException) as e:
        logging.error(e.value)
        return False
