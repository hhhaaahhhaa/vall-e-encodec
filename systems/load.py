from typing import Type

from . import bart_vc_ar
from . import t5_bart_vc_ar


SYSTEM_ENC_DEC = {
    "bart/bart": (bart_vc_ar.System, bart_vc_ar.DataModule),
    "t5-base/bart": (t5_bart_vc_ar.System, t5_bart_vc_ar.DataModule),
}


SYSTEM = {
    **SYSTEM_ENC_DEC,
}


def get_system(system_name: str):
    return SYSTEM[system_name][0]


def get_datamodule(system_name: str):
    return SYSTEM[system_name][1]
