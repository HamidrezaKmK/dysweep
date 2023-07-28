from dataclasses import fields
from dataclasses import dataclass, is_dataclass
from typing import get_type_hints, get_origin, get_args
import typing as th

def parse_dictionary_onto_dataclass(input_dict: dict, dataclass_type):
    # NOTE: jsonargparse has probably implemented something similar to this
    all_fields = [f.name for f in fields(dataclass_type)]
    all_dataclass_type_hints = get_type_hints(dataclass_type)
    instances = {}
    for key, val in input_dict.items():
        if key not in all_fields:
            raise ValueError(f"Key {key} not in dataclass fields")
        typehint = None if key not in all_dataclass_type_hints else all_dataclass_type_hints[key]
        # check if typehint is a subscripted generic, in that case, just ignore it
        if typehint is not None:
            def is_optional(field):
                return th.get_origin(field) is th.Union and \
                    type(None) in th.get_args(field)
            while is_optional(typehint):
                for args in th.get_args(typehint):
                    if args is not type(None):
                        typehint = args
                        break
            if getattr(typehint, "__origin__", None) is th.Dict:
                if not isinstance(val, dict):
                    raise ValueError(f"Value {val} for key {key} is not a dictionary")
                instances[key] = val
            elif not hasattr(typehint, "__origin__"):
                if is_dataclass(typehint):
                    instances[key] = parse_dictionary_onto_dataclass(val, typehint)
                else:
                    try:
                        instances[key] = typehint(val)
                    except Exception as e:
                        raise ValueError(f"Value {val} for key {key} is not of type {typehint}")
            else:
                instances[key] = val
        else:
            instances[key] = val
            
    return dataclass_type(**instances)
