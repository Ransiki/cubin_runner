import re
import uuid
import json
import inspect
import datetime
import warnings
from enum import Enum
from typing import List, Union, Any, Dict, get_type_hints
import pydantic
from pydantic.json import pydantic_encoder


alphanumeric_only_re = re.compile(r"[^a-z0-9\.]+")


class NVDataFlowFormatWarning(UserWarning):
    pass


class UnknownEnumValueWarning(UserWarning):
    pass


class Text(str):
    """
    Maps to an ES `text` field, instead of `keyword`.
    https://www.elastic.co/guide/en/elasticsearch/reference/current/text.html
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if v is None or (isinstance(v, list) and not v):
            return None

        # Keep lists, but convert its items to string
        if isinstance(v, list):
            return [cls(str(v_)) for v_ in v]

        return cls(str(v))


class Extra(Dict[str, Any]):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        # Convert keys to string
        return cls(
            {str(k): v_ for k, v_ in v.items() if k is not None and v_ is not None}
        )


class LooseEnum(str, Enum):
    """Enum that allows unknown values, with a warning."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, values, field=None, **kwargs):
        if not isinstance(v, str):
            raise TypeError("string required")

        # Try to get the Enum from value; if it doesn't exist, use the string itself
        if type(v) is str:
            v = next(
                (
                    f
                    for f in cls
                    if (
                        # strict test
                        f.value == v
                        # ignoring case
                        or f.value.lower() == v.lower()
                        # compare using only alphanumeric and `.` characters
                        or alphanumeric_only_re.sub("", f.value.lower())
                        == alphanumeric_only_re.sub("", v.lower())
                    )
                ),
                v,
            )

        if not isinstance(v, cls) or v not in cls:
            warnings.warn(
                'Invalid value "{}" in field "{}". Accepted values: {}.'.format(
                    v, field.alias, ", ".join(f for f in cls)
                ),
                UnknownEnumValueWarning,
            )
        return v

    def __str__(self):
        return self.value

    def __repr__(self):
        return str.__repr__(self.value)


class KeyValueDict(Dict[str, str]):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        # Convert keys to string
        return cls(
            {str(k): str(v_) for k, v_ in v.items() if k is not None and v_ is not None}
        )


class FlatDict(Dict[str, str]):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return cls(
            {str(k): str(v_) for k, v_ in v.items() if k is not None and v_ is not None}
        )


NVDATAFLOW_NAMING_WARNING = "Field {} does not conform to the NVDataFlow naming scheme by not containing a prefix indicating the type, and may be indexed with the wrong type. Check the naming rules in confluence and rename the field to contain a type prefix: https://confluence.nvidia.com/display/nvdataflow/NVDataFlow#NVDataFlow-DataFieldNamingRules."

NVDATAFLOW_PREFIX_MAPPING = {
    Text: "text_",
    KeyValueDict: "flat_",
    FlatDict: "flat_",
    str: "s_",
    int: "l_",
    float: "d_",
    bool: "b_",
    datetime.datetime: "ts_",
    LooseEnum: "s_",
}

NVDATAFLOW_PREFIXES = (
    "l_",
    "d_",
    "b_",
    "ts_",
    "s_",
    "ni_",
    "ns_",
    "text_",
    "rl_",
    "rd_",
    "rts_",
    "flat_",
    "nested_",
)


def encoder(obj):
    # Assume all dates are in UTC
    if isinstance(obj, (datetime.datetime, datetime.date)):
        ret = f"{obj.astimezone(tz=datetime.timezone.utc).replace(tzinfo=None).isoformat()}Z"
    else:
        ret = pydantic_encoder(obj)
    return ret


def get_type_prefix(type_):
    return next(
        (
            NVDATAFLOW_PREFIX_MAPPING.get(t, "")
            for t in [type_] + list(getattr(type_, "__bases__", []))
            if NVDATAFLOW_PREFIX_MAPPING.get(t, "")
        ),
        "",
    )


class Base(pydantic.BaseModel):
    # mapping of alias -> field
    _aliases = {}

    # mapping of prefix type -> fields, to store as duplicate fields in ES
    # eg: {Text: [string_field, ...]}
    _duplicate_as = {}

    class Config:
        extra = "ignore"
        validate_assignment = True

    def __init__(self, *args, **kwargs):
        if self.__class__ == Base:
            raise TypeError("Can't instantiate abstract class Base.")

        cls = self.__class__
        model_annotations = get_type_hints(cls)

        for name in list(kwargs.keys()):
            # Support aliases by renaming them in kwargs
            if name in self._aliases:
                kwargs[self._aliases[name]] = kwargs.pop(name)

            # Auto-alias fieldnames prefixed with types,
            # _only_ if 1) the non-prefixed field exists, and 2) the type matches the prefix
            # Eg. s_name => name, if `name: str` exists
            if not hasattr(self, name):
                _prefix = next(
                    (p for p in NVDATAFLOW_PREFIXES if name.startswith(p)), None
                )
                if _prefix:
                    new_name = name[len(_prefix) :]
                    type_ = self._get_type(new_name)
                    type_prefix = get_type_prefix(type_)

                    if new_name in model_annotations and type_prefix == _prefix:
                        kwargs[new_name] = kwargs.pop(name)

        super().__init__(*args, **kwargs)
    
    def __getattr__(self, name):
        # Retrieve aliases as well
        if name in self._aliases:
            return getattr(self, self._aliases[name])

        # Try unprefixed field
        return self._get_unprefixed_attr(name)

    def __setattr__(self, name, value):
        try:
            return super().__setattr__(name, value)
        except ValueError:
            # Check aliases
            if name in self._aliases:
                return super().__setattr__(self._aliases[name], value)
            else:
                raise
    
    @classmethod
    def _get_type(cls, name):
        model_annotations = get_type_hints(cls)
        type_ = model_annotations.get(name)

        if type_ is None:
            # potentially a property, so try pulling typing data
            try:
                model_annotations = get_type_hints(getattr(getattr(cls, name), "fget"))
                type_ = model_annotations["return"]
            except (AttributeError, KeyError):
                pass  # Was not a property

        # Optional fields are type Union(type, None)
        # List fields also contain the type in `__args__[0]`
        while getattr(type_, "__origin__", None) in (Union, List, list):
            type_ = type_.__args__[0]
        return type_
    
    @classmethod
    def _split_prefix(cls, name):
        prefix, without_prefix = next(
            (
                (name[: len(p)], name[len(p) :])
                for p in NVDATAFLOW_PREFIXES
                if name.startswith(p)
            ),
            ("", name),
        )
        return prefix, without_prefix
    
    @classmethod
    def _get_unprefixed_attr_name(cls, name):
        prefix, without_prefix = cls._split_prefix(name)
        type_ = cls._get_type(without_prefix)
        type_prefix = get_type_prefix(type_)
        if prefix and type_ and type_prefix == prefix:
            return without_prefix
        return name
    
    def _get_unprefixed_attr(self, name):
        unprefixed_name = self._get_unprefixed_attr_name(name)
        return self.__getattribute__(unprefixed_name)

    def update(self, *args, **kwargs):
        _new_fields = {}
        for _args in args:
            if isinstance(_args, dict):
                _new_fields.update(_args)
        if kwargs:
            _new_fields.update(kwargs)
        for k, v in _new_fields.items():
            setattr(self, k, v)
    
    # @pydantic.validator("_type", always=True, check_fields=False)
    # def check_type(cls, v):
    #     # For subclasses, ensure there's a `_type`
    #     if v:
    #         raise ValueError("Can't specify a _type for instances")

    @pydantic.validator("id", pre=True, always=True, check_fields=False)
    def set_default_id(cls, v):
        return v or str(uuid.uuid4())

    @pydantic.validator("*", pre=True)
    def disallow_empty_string(cls, v):
        return None if v == "" else v

    @classmethod
    def _dict_encoder(cls, field: str, obj: Any, type_: Any):
        """
        Format a field/value pair to conform to the NVDataFlow format.

        Automatically adds the proper prefix based on the value type.
        :param field: field name
        :param obj: value of the field
        :param type_: type of the value; used to add a prefix to the field
        :return: a tuple with the new field and value
        """
        if not type_:
            return field, obj

        # Apply prefix (look in parent classes as well)
        prefix = get_type_prefix(type_)
        field = f"{prefix}{field}"

        if obj is None or (isinstance(obj, list) and not obj):
            return field, obj

        if field == "extra" and isinstance(obj, dict):
            for f in obj.keys():
                if not f.startswith(NVDATAFLOW_PREFIXES):
                    warnings.warn(
                        NVDATAFLOW_NAMING_WARNING.format(f"extra.{f}"),
                        NVDataFlowFormatWarning,
                    )

        # Process Dict; process values according to their type
        if getattr(type_, "__origin__", None) in (dict, Dict):
            val_type = type_.__args__[1]
            is_list = getattr(val_type, "__origin__", Dict) in (list, List)
            while getattr(val_type, "__origin__", None) in (Union, List, list):
                val_type = val_type.__args__[0]

            if inspect.isclass(val_type) and issubclass(val_type, Base):
                obj = {
                    k: [val_type.dict(vv, nvdataflow=True) for vv in v]
                    if is_list
                    else val_type.dict(v, nvdataflow=True)
                    for k, v in obj.items()
                }

        elif inspect.isclass(type_):
            if issubclass(type_, KeyValueDict):
                obj = [
                    {"s_key": k, "s_value": v}
                    for k, v in obj.items()
                    if k is not None and v is not None
                ]

            # Nested models?
            elif issubclass(type_, Base):
                if isinstance(obj, list):
                    obj = [type_.dict(o, nvdataflow=True) for o in obj]
                else:
                    obj = type_.dict(obj, nvdataflow=True)

        return field, obj

    @classmethod
    def _dict_decoder(cls, field: str, obj: Any, custom_classes: Dict[Any, Any] = None):
        """
        Decode an NVDataFlow field/value pair into the corresponding types defined in this class.

        :param field: field name (possibly with an NVDataFlow prefix)
        :param obj: field value
        :param custom_classes: dict of "original type" to "custom type" to rebuild the object with different classes
                               eg. if the current model contains a field with type Workload, but we want the decoded
                               object to have the type CustomWorkload, use `custom_classes={Workload: CustomWorkload}`
        :return: the decoded field/value pair
        """
        custom_classes = custom_classes or {}
        orig_field = field
        _prefix, field = cls._split_prefix(field)
        obj_is_list = isinstance(obj, list)

        # List of @property attributes
        # TODO: move to separate @classmethod
        attributes = {prop: getattr(cls, prop) for prop in dir(cls)}
        properties = [
            name
            for name, attribute in attributes.items()
            if isinstance(attribute, property)
            and name not in ("__values__", "fields")
            and not name.startswith("_")
        ]

        if orig_field in properties:
            return orig_field, obj

        _type = cls._get_type(field)

        # Is `obj` a nested type? If so, decode it by calling this method recursively
        if inspect.isclass(_type) and issubclass(_type, Base):
            # custom class (eg. CustomWorkload)?
            _type = custom_classes.get(_type) or _type

            # Iterate obj as if it was a list, but transform it back to the original type at the end
            # (either a list or an obj)
            obj_data = {}
            obj_list = obj if obj_is_list else [obj]
            obj = [] if obj_is_list else None
            for _obj in obj_list:
                for sub_f, sub_v in (_obj or {}).items():
                    f, v = _type._dict_decoder(
                        sub_f, sub_v, custom_classes=custom_classes
                    )
                    obj_data[f] = v
                decoded_obj = _type(**obj_data)
                # Convert back to either a list or an obj
                if obj_is_list:
                    obj.append(decoded_obj)
                else:
                    obj = decoded_obj

        else:
            if _prefix == "nested_":
                obj = {
                    item["s_key"]: item["s_value"]
                    for item in obj
                    if all(f in item for f in ("s_key", "s_value"))
                }
            if _prefix == "flat_":
                new_obj = {}
                for k, v in obj.items():
                    _pre, k = cls._split_prefix(k)
                    new_obj[k] = v
                obj = new_obj

            # Timestamp; remove timezone info
            elif _prefix == "ts_" and isinstance(obj, str) and obj.endswith("Z"):
                obj = obj[: len(obj) - 1]

        return field, obj

    def dict(self, nvdataflow=False, **kwargs):
        if not self:
            return None

        if not nvdataflow:
            return super().dict(**kwargs)

        # NVDataFlow

        # Pre-process fields and types
        cls = self.__class__
        res = {}
        to_process = []

        # Attributes
        for field in super().dict(**kwargs).keys():
            if field.startswith("_"):
                continue

            value = getattr(self, field, None)
            type_ = self._get_type(field)
            to_process.append((field, value, type_))

        # Properties
        attributes = {prop: getattr(cls, prop) for prop in dir(cls)}
        properties = [
            name
            for name, attribute in attributes.items()
            if isinstance(attribute, property)
            and name not in ("__values__", "fields")
            and not name.startswith("_")
        ]
        for field in properties:
            value = getattr(self, field, None)
            type_ = self._get_type(field)
            to_process.append((field, value, type_))

        # Queue duplicates as well (look in all parent classes)
        for clss in [self.__class__] + list(
            mro for mro in self.__class__.__mro__ if issubclass(mro, Base)
        ):
            for type_ in getattr(clss, "_duplicate_as", {}):
                for field in clss._duplicate_as.get(type_, []):
                    try:
                        value = getattr(self, field, None)
                        if hasattr(type_, "validate"):
                            value = type_.validate(value)
                        to_process.append((field, value, type_))
                    except TypeError:
                        pass

        # Now process the queued fields
        for (field, value, type_) in to_process:
            if field.startswith("_"):
                continue

            field, value = self._dict_encoder(field, value, type_)

            # Skip null values and empty dicts/lists
            if value is None or (isinstance(value, (list, dict)) and not value):
                continue

            # Warn when additional (extra) fields do not conform with the NVDataFlow naming scheme
            if not type_ and not field.startswith(NVDATAFLOW_PREFIXES):
                warnings.warn(
                    NVDATAFLOW_NAMING_WARNING.format(field), NVDataFlowFormatWarning
                )

            res[field] = value

        return res

    def json(self, nvdataflow=False, **kwargs):
        return json.dumps(
            self.dict(nvdataflow=nvdataflow, **kwargs), default=encoder, **kwargs
        )

    @classmethod
    def from_dict(cls, data, nvdataflow=True, **kwargs):
        """
        Create a base model object based on a dict.

        This is the reverse method to :py:func:`.dict`
        """
        model_params = {}
        for field, value in data.items():
            f, v = cls._dict_decoder(field, value, **kwargs)
            model_params[f] = v

        return cls(**model_params)

    @classmethod
    def from_json(cls, data, nvdataflow=True, **kwargs):
        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):
            return cls.from_dict(data, nvdataflow=nvdataflow, **kwargs)
