import tempfile
import os
import copy
import json
import shutil
import pickle
import logging
import operator
from enum import Enum
from functools import reduce
from typing import Any, Dict, List, Optional, Union

import requests

from .base import Base, encoder
from .models import Record, PerformanceResult, Workload, MetricRecord
from .utils.swiftstack import test_access, upload_files, construct_url

logger = logging.getLogger(__name__)
ES_INDEX_PREFIX = "compute_arch-cnexus_perfdb_v2"
NVDATAFLOW_URL = "http://gpuwa.nvidia.com/dataflow"


class Env(Enum):
    """
    PerfDB environment, which decides on which index data will be stored.

    Use :py:attr:`~DEV` during development, :py:attr:`~TEST` for automated tests,
    and :py:attr:`~PROD` for production data.
    """

    PROD = "production"
    DEV = "development"
    TEST = "testing"


class Api:
    _env = Env.DEV
    _queued = []

    def __init__(self, env: Env = Env.DEV):
        """
        :param env: Environment
        """
        self._env = env
        self._queued = []

    def _push_data(self, index, data):
        if not data or not index:
            return

        url = f"{NVDATAFLOW_URL}/{index}/posting"
        res = requests.post(
            url,
            headers={"Content-Type": "application/json", "Accept-Charset": "UTF-8"},
            data=data,
        )
        try:
            res.raise_for_status()
        except requests.HTTPError:
            logger.error("Error posting to NVDataFlow: %s", res.text or "uknown error")
            raise

    def _get_index(self, data: Record, env=None):
        if not data:
            raise ValueError("No data")

        return "{prefix}-{env}-{client}-{result_type}".format(
            prefix=ES_INDEX_PREFIX,
            env=(env or self._env).value,
            client=data.client or "unknown",
            result_type=data._type or "unknown",
        )

    def _group_items_by_index(self, data: List[Record]):
        """Return a dict of lists of items, index by the ES index.

        Takes a list of models and organizes them into buckets, depending on the contents.
        """
        # Convert single item to a list
        if not isinstance(data, list):
            data = [data]

        to_push = {}  # {es_index: [items]}
        for item in data:
            index = self._get_index(item, env=self._env)
            if not index:
                continue

            if index not in to_push:
                to_push[index] = []

            to_push[index].append(item)
        return to_push

    @classmethod
    def _to_json_list(cls, record_dict_list, max_size=None):
        """
        Convert a list of records (already converted to dicts) to a list of json-encoded requests.

        NVDataFlow has request size limits, so each list item is a json request ready to be pushed;
        each list item will be a json-encoded list of runs, with a `max_size` (eg. 1MB).

        Return example:
        [
          "[{...run1},{...run2}]",
          "[{...run3},{...run4},{...run5}]",
          ...
        ]
        """
        result_list = []

        record_list = []
        record_list_size = len("[]".encode("utf-8"))  # empty list size
        for record_dict in record_dict_list:
            record_json = json.dumps(record_dict, default=encoder)
            record_json_size = len(record_json.encode("utf-8"))

            if max_size and (record_list_size + record_json_size) > max_size:
                result_list.append("[" + ",".join(record_list) + "]")
                record_list = []
                record_list_size = len("[]".encode("utf-8"))  # empty list size

            record_list.append(record_json)
            record_list_size += record_json_size + 1

        # Any result list item pending? finish it
        if record_list:
            result_list.append("[" + ",".join(record_list) + "]")

        return result_list

    # Convert nested types to flattened for anna-operator
    # TODO: remove once flattened types enabled for all indices
    def _fix_flat_types(self, data):
        if isinstance(data, dict):
            data = [data]

        for doc in data:
            fields_to_add = {}
            fields_to_remove = []
            for field, value in doc.items():
                if field.startswith("nested_"):
                    new_field = "flat_" + field[7:]
                    # can't modify a dict in the middle of an iteration, so queue it for later
                    fields_to_add[new_field] = value
                    fields_to_remove.append(field)

            for field, value in fields_to_add.items():
                doc[field] = value
            for field in fields_to_remove:
                doc.pop(field, None)

        return data

    def _prepare_data(self, data):
        """
        Prepare the data by converting all records to JSON and organizing them into buckets by index.

        Returns: dict of index -> list of requests (of JSON-encoded docs)
        """
        if not data:
            raise ValueError("No data to push.")

        if not isinstance(data, (Record, list)) or (
            isinstance(data, list) and any(not isinstance(r, Record) for r in data)
        ):
            raise ValueError(
                "Data not recognized, pass one or a list of Record objects."
            )

        if not isinstance(data, list):
            data = [data]

        # extract `PeformanceResult.metrics` into separate records and fill in their result IDs
        extra_metric_records = []
        for item in data:
            if isinstance(item, PerformanceResult) and len(item.metrics):
                metrics = [m for m in item.metrics if isinstance(m, MetricRecord)]

                for record_idx, metric_record in enumerate(metrics):
                    if not metric_record.result_id:
                        metric_record.result_id = item.id
                    if metric_record.seq_id is None:
                        metric_record.seq_id = record_idx

                    extra_metric_records.append(metric_record)

                item.has_metrics = len(metrics) > 0
                item.metrics = []

        data += extra_metric_records

        ret = {}
        items_per_index = self._group_items_by_index(data)
        for index, items in items_per_index.items():
            if items:
                items_dict_list = [d.dict(nvdataflow=True) for d in items]
                if "anna-operator" in index:
                    items_dict_list = self._fix_flat_types(items_dict_list)

                # Split into ~1MB requests
                req_list = self._to_json_list(
                    items_dict_list, max_size=(0.95 * 1024 * 1024)
                )
                for req_data in req_list:
                    if index not in ret:
                        ret[index] = []

                    ret[index].append(req_data)
            else:
                logger.warning(f"No items to push for index {index}")
        return ret

    def _update_record(
        self,
        records: Union[Record, List[Record], dict, List[dict]],
        update_callback=(lambda d: d),
        operation_name="update records",
    ):
        """
        Update one or a list of either :py:class:`Records <.models.Record>` or JSON-encoded (eg. retrieved from a query) Records.

        Ideally the records passed were already retrieved with :py:attr:`~query`, and will already contain the ElasticSearch document id and index (`_id` and `_index` fields); otherwise a new query will be made to retrieve them.
        """
        if not isinstance(records, list):
            records = [records]

        ids_to_invalidate = (
            {}
        )  # map of (_index => (_id, ts_created)) of ES docs to invalidate

        records_to_query = {}  # map of (client => id)
        for record in records:
            # If actual records were passed, we need to retrieve the ES _index and _id fields from ES
            if isinstance(record, Record):
                # Are the client/id fields present? If not we can't query the doc from ES
                if any(not getattr(record, f, None) for f in ("client", "id")):
                    raise Exception(
                        "Unable to %s: some records are missing either the ID or the client",
                        operation_name,
                    )

                if not records_to_query.get(record.client):
                    records_to_query[record.client] = []

                records_to_query[record.client].append(record.id)

            # If this is a json-encoded Record, check if we have _index and _id; if not, query them as well
            elif isinstance(record, dict):
                if all(record.get(f) for f in ("_index", "_id")) and record.get(
                    "_source", {}
                ).get("ts_created"):
                    # Best case scenario: _index, _id and ts_created present, don't need to query for this record
                    if not ids_to_invalidate.get(record["_index"]):
                        ids_to_invalidate[record["_index"]] = []
                    ids_to_invalidate[record["_index"]].append(
                        (record["_id"], record["_source"]["ts_created"])
                    )
                else:
                    if record.get("_source"):
                        record = record["_source"]

                    # Are the client/id fields present? If not we can't query the doc from ES
                    if any(not record.get(f) for f in ("s_client", "s_id")):
                        raise Exception(
                            "Unable to %s: some records are missing either the ID or the client",
                            operation_name,
                        )

                    if not records_to_query.get(record["s_client"]):
                        records_to_query[record["s_client"]] = []
                    records_to_query[record["s_client"]].append(record["s_id"])

        # Query any documents for the _index/_id if needed
        if records_to_query:
            for (record_client, record_ids) in records_to_query.items():
                res = self.query(
                    {
                        "size": 10,
                        "query": {
                            "bool": {"filter": [{"terms": {"s_id": record_ids}}]}
                        },
                    },
                    record_client,
                )
                hits = res.get("hits", {}).get("hits", [])
                if not hits:
                    raise Exception(
                        "Unable to %s: no documents retrieved from ES.", operation_name
                    )

                for hit in hits:
                    if not ids_to_invalidate.get(hit["_index"]):
                        ids_to_invalidate[hit["_index"]] = []
                    ids_to_invalidate[hit["_index"]].append(
                        (hit["_id"], hit["_source"]["ts_created"])
                    )

        # We now have all the information we need, so invalidate the docs
        for (_index, docs) in ids_to_invalidate.items():
            # NVDataFlow requires that we use the original index, _not_ the real one that they append a suffix/prefix to.
            # So extract the prefix (`df-`) and the shard (shard, whatever is after the last `-`), which we'll also need to pass
            _shard = _index[_index.rfind("-") + 1 :]
            _index = _index[_index.find("-") + 1 : _index.rfind("-")]
            body = json.dumps(
                [
                    update_callback(
                        {
                            "_id": _id,
                            "_shard": _shard,
                        }
                    )
                    for (_id, ts_created) in docs
                ]
            )
            self._push_data(_index, body)

    def _upload_assets(self, records: List[Record]):
        if not records:
            raise ValueError("No data to push.")

        if not isinstance(records, (Record, list)) or (
            isinstance(records, list)
            and any(not isinstance(r, Record) for r in records)
        ):
            raise ValueError(
                "Data not recognized, pass one or a list of Record objects."
            )

        if not isinstance(records, list):
            records = [records]

        for record in filter(
            lambda record: isinstance(record, PerformanceResult), records
        ):
            container = record.client
            timestamp = record.timestamp
            if record.run_name:
                id = record.run_name
            else:
                id = record.id

            assets = list(filter(lambda a: a.remote, record.assets))
            if assets:
                with tempfile.NamedTemporaryFile(mode="w+") as metadata_file:
                    remote_folder = f"{timestamp.strftime('%Y-%m-%d')}/{id}"
                    metadata = {
                        "_index": self._get_index(record),
                        "s_run_name": record.run_name,
                        "s_id": record.id,
                    }
                    metadata_file.write(json.dumps(metadata, indent=2))
                    metadata_file.flush()

                    to_upload = [
                        {
                            "dest": f"{remote_folder}/es_metadata.json",
                            "src": metadata_file.name,
                        }
                    ]

                    for asset in assets:

                        remote_path = f"{remote_folder}/{asset.name}"
                        to_upload.append(
                            {
                                "dest": remote_path,
                                "src": asset.path,
                            }
                        )
                        asset.path = remote_path
                        asset.url = construct_url(remote_path, container)

                    status = upload_files(to_upload, container)
                    for asset, s in zip(assets, status[:-1]):
                        asset.upload_status = s

    # Public API:

    def is_accessible(
        self, records: bool = True, assets: bool = True, timeout: int = 10
    ) -> bool:
        """
        Tests access to the target servers, so we can check if we will be able to push data.

        :param records: able to upload records?
        :param assets: able to upload assets?
        :return: whether we can access the required servers
        """
        # requests.head(NVDATAFLOW_URL)
        try:
            return (
                records is False or requests.head(NVDATAFLOW_URL).status_code == 404
            ) and (assets is False or test_access(timeout=timeout))
        except (requests.exceptions.RequestException):
            pass
        return False

    def serialize(
        self, records: Union[Record, List[Record]], to_file: str = None
    ) -> bytes:
        """
        Serialize one or multiple :py:class:`Records <.models.Record>`.

        If `to_file` is specified, the serialized data is also written to the corresponding file.
        """
        if not isinstance(records, list):
            records = [records]

        data = pickle.dumps({"records": records})
        if to_file:
            with open(to_file, "wb") as outfile:
                outfile.write(data)

        return data

    def deserialize(self, data: bytes = None, from_file: str = None) -> List[Record]:
        """
        Deserialize one or multiple :py:class:`Records <.models.Record>` from either a string or a file.

        If `data` is specified, it will try to deserialize the records from the given data.
        If `from_file=path` is specified, it will try to deserialize the records from the corresponding file.
        """
        if not data and from_file:
            with open(from_file, "rb") as infile:
                data = infile.read()

        loaded_data = pickle.loads(data)
        records = loaded_data.get("records")
        return records

    def create_archive(
        self, records: Union[Record, List[Record]], to_file: str
    ) -> None:
        """
        Create an archive with one or multiple :py:class:`Records <.models.Record>`, including assets.

        Assets will be copied over from their original destinations, and the filenames changed to `<record index>_<asset index>_<original filename>`.
        Eg. the second asset of the third record `myasset.json` will be `2_1_myasset.json`.
        """
        if not isinstance(records, list):
            records = [records]

        # Trim file extension from destination file, since `make_archive` will add it.
        if to_file.endswith(".zip"):
            to_file = to_file[0:-4]

        # Create temp dir to create the archive
        with tempfile.TemporaryDirectory() as tmpdirname:
            # For every record, move the assets to the directory we'll zip up
            for r_i, record in enumerate(records):
                for a_i, asset in enumerate(getattr(record, "assets", [])):
                    if not os.path.exists(asset.path):
                        raise Exception(f"Asset file {asset.path} not found.")

                    # Copy the file over and update the path in the record
                    basepath, filename = os.path.split(asset.path)
                    new_file_name = f"{r_i}_{a_i}_{filename}"
                    fpath = os.path.join(tmpdirname, new_file_name)
                    shutil.copy(asset.path, fpath)
                    asset.path = new_file_name

            srecords_path = os.path.join(tmpdirname, "records.bin")
            self.serialize(records, to_file=srecords_path)
            shutil.make_archive(to_file, "zip", tmpdirname)

    def deserialize_archive(self, from_file: str, to_dir: str):
        """
        Deserialize one or multiple records previously archived with :py:class:`create_archive <.create_archive>`.
        """
        if not os.path.exists(from_file):
            raise Exception(f"Archive {from_file} not found.")

        shutil.unpack_archive(from_file, to_dir, "zip")
        srecords_path = os.path.join(to_dir, "records.bin")
        return self.deserialize(from_file=srecords_path)

    def push_archive(self, from_file: str) -> None:
        """
        Upload one or multiple records previously archived with :py:class:`create_archive <.create_archive>`.
        """
        if not os.path.exists(from_file):
            raise Exception(f"Archive file {from_file} not found.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            shutil.unpack_archive(from_file, tmpdirname, "zip")
            fpath = os.path.join(tmpdirname, "records.bin")
            with open(fpath, "rb") as f:
                srecords = f.read()

            records = self.deserialize(
                srecords, from_file=os.path.join(tmpdirname, "records.bin")
            )

            for record in records:
                for asset in getattr(record, "assets", []):
                    asset.path = os.path.join(tmpdirname, asset.path)

            self.push(records)

    def queue(self, records: Union[Record, List[Record]]):
        """
        Queue one or multiple records, to be pushed later using :py:attr:`~push_queued`.

        :param records: records to be queued
        """
        if isinstance(records, Record):
            records = [copy.deepcopy(records)]
        elif isinstance(records, (list)):
            records = [copy.deepcopy(r) for r in records if isinstance(r, Record)]
        else:
            logger.warning('Can only queue one or multiple "Record" objects.')

        if not len(records):
            logger.warning("No records to queue.")
            return

        self._queued += records

    def push_queued(self):
        """Push all previously queued records to NVdataFlow, and flush the queue."""
        self.push(self._queued)
        self._queued = []

    def push(self, records: Union[Record, List[Record]]):
        """
        Push one or a list of :py:class:`Records <.models.Record>` to NVDataFlow.

        :param records: records to be pushed
        """

        self._upload_assets(records)
        data_to_push = self._prepare_data(records)
        for index, items in data_to_push.items():
            for data in items:
                self._push_data(index, data)

    def invalidate_record(self, records: Union[Record, List[Record], dict, List[dict]]):
        """
        Invalidate one or a list of either :py:class:`Records <.models.Record>` or JSON-encoded (eg. retrieved from a query) Records.

        Ideally the records passed were already retrieved with :py:attr:`~query`, and will already contain the ElasticSearch document id and index (`_id` and `_index` fields);
        otherwise a new query will be made to retrieve them.
        """
        return self._update_record(
            records,
            update_callback=(lambda d: {**d, "b_invalid": True}),
            operation_name="invalidate records",
        )

    def query_similar_records(
        self,
        similar_to: Record,
        match_fields: Optional[str] = None,
        custom_classes: Dict[Base, Base] = None,
        limit=10,
    ) -> List[Record]:
        """
        Query for PerfDB records similar to a given record, based on a given set of fields.
        Client ID and record type are matched automatically.

        The returned records have the same type as the original record.

        Eg. Query for all records that match the "network" and "batch size" of record `resnet50_bs16_record`:
            `query_similar_records(resnet50_bs16_record, match_fields=["workload.s_network", "workload.d_batch_size"])`

        :param similar_to: the record to compare against
        :param match_fields: path to the field(s) to use for comparison;
                               eg. `["workload.s_network", "workload.d_batch_size"]`
        :param custom_classes: custom classes to use when rebuilding the retrieved records
                               the top-level Record and the Workload classes are matched automatically with the original record
        :param limit: number of records to return; defaults to 10
        :return: list of matched records (with the same type as the original record)
        """
        record = similar_to.dict(nvdataflow=True)
        custom_classes = custom_classes or {}
        match_fields = match_fields or []

        if not custom_classes.get(Record):
            custom_classes[Record] = similar_to.__class__
        if not custom_classes.get(Workload) and isinstance(
            getattr(similar_to, "workload"), Workload
        ):
            custom_classes[Workload] = similar_to.workload.__class__

        # Build the query filter based on the matched fields of the original record
        query_filter = []
        for f in match_fields:
            value = None
            try:
                value = reduce(operator.getitem, f.split("."), record)
            except KeyError:
                pass
            query_filter.append({"term": {f: value}})

        query = {
            "size": limit,
            "query": {"bool": {"filter": query_filter}},
            "sort": [{"ts_created": {"order": "desc"}}],
        }
        res = self.query(
            query, client=record.get("s_client"), result_type=record.get("s_type")
        )
        hits = res.get("hits", {}).get("hits", [])

        # Convert hits back to Record objects
        for i, hit in enumerate(hits):
            hits[i] = custom_classes[Record].from_dict(
                hit.get("_source"), custom_classes=custom_classes
            )
        return hits

    def query(
        self,
        query: Dict[Any, Any],
        client: Optional[str] = "*",
        result_type: Optional[str] = "*",
        env: Optional[Union[Env, str]] = None,
        date_index="*",
    ):
        """
        Query data from PerfDB.

        Query input must be a valid ElasticSearch query. Result will be an ElasticSearch query response.

        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html

        Example usage:

        .. code-block:: python

           api.query(
               {
                   "size": 10,
                   "query": {
                       "bool": {
                           "filter": [
                               {"term": {"session.workload.result.s_network": "ResNet50"}}
                           ]
                       }
                   },
                   "sort": [
                       {"ts_created": {"order": "desc"}}
                   ]
               },
               client="flow.dnn",
           )

        :param query: query using ElasticSearch's query format
        :param client: filter by client ID, defaults to `*` (all clients)
        :param result_type: filter by type of result (eg. `network` for Network results, defaults to `*` (all results)
        :param env: Environment, defaults to :py:class:`PROD <cnexus_perfdb.api.Env.PROD>`
        :return: ElasticSearch query result
        """
        if not env:
            env = self._env
        res = requests.post(
            f"http://gpuwa.nvidia.com/elasticsearch/df-{ES_INDEX_PREFIX}-{env.value}-{client}-{result_type}-{date_index}/_search",
            json=query,
        )
        res.raise_for_status()
        return res.json()
