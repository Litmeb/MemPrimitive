import memengine
from memengine.config.Config import MemoryConfig


config = MemoryConfig(
    {
        "global_config": {"usable_gpu": ""},
        "storage": {},
        "store": {},
        "recall": {
            "truncation": {"method": "LMTruncation", "mode": "word", "number": 200},
            "utilization": {
                "method": "ConcateUtilization",
                "prefix": "",
                "suffix": "",
                "list_config": {"index": True, "sep": "\n"},
                "dict_config": {"key_value_sep": ": ", "key_format": "%s", "item_sep": "\n"},
            },
        },
        "display": {
            "method": "ScreenDisplay",
            "key_value_sep": ": ",
            "key_format": "%s",
            "item_sep": "\n",
            "prefix": "=== %s ===",
            "suffix": "",
        },
    }
)

memo = memengine.FUMemory(config)