from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.utils.data import Dataset
import re
from collections import defaultdict


DATA_DIR = ...
MIMIC_2_DIR = ...
MIMIC_3_DIR = ...
ICD_50_RANK = ...

HEADERS = {
    "family history": "fam/sochx",
    "social history": "fam/sochx",
    "history of present illness": "genhx",
    "past medical history": "pastmedicalhx",
    "chief complaint": "cc",
    "past surgical history": "pastsurgical",
    "allergies": "allergy",  # "allergies"
    "review of systems": "ros",
    "medications": "medications",
    "assessment": "assessment",
    "exam": "exam",
    "diagnosis": "diagnosis",
    "diagnoses": "diagnosis",
    "discharge diagnosis": "diagnosis",
    "discharge diagnoses": "diagnosis",
    "disposition": "disposition",
    "plan": "plan",
    "emergency department course": "edcourse",
    "immunizations": "immunizations",
    "imaging": "imaging",
    "gynecologic history": "gynhx",
    "procedures": "procedures",
    "other history": "other_history",
    "labs": "labs",
}


def proc_text(text):
    text = text.lower().replace("\n", " ").replace("\r", " ")
    text = re.sub("dr\.", "doctor", text)
    text = re.sub("m\.d\.", "doctor", text)
    text = re.sub("admission date:", "", text)
    text = re.sub("discharge date:", "", text)
    text = re.sub("--|__|==", "", text)
    return re.sub(r"  +", " ", text)


def get_headersandindex(input_str):
    input_str = input_str.lower()
    headers_to_select = [f"{k}:" for k in HEADERS.keys()]

    strs = input_str.split("\n")
    headers = []
    for str_tmp in strs:
        str_tmp = str_tmp.strip()
        if len(str_tmp) > 0 and str_tmp[-1] == ":":
            headers.append(str_tmp)
    headers_pos = []
    last_index = 0
    for header in headers:
        starts = last_index + input_str[last_index:].index(header)
        last_index = starts + len(header)
        headers_pos.append((header, starts))
    headers_pos += [("end:", len(input_str))]
    counta = 0
    finals = []
    while counta < len(headers_pos) - 1:
        (header, starts) = headers_pos[counta]
        if header in headers_to_select:
            finals.append(
                (HEADERS[header.replace(":", "")], starts, headers_pos[counta + 1][1])
            )  # (section headername, start of section, end of section)
        counta += 1
    return finals


def get_subnote(input_str, headers_pos):
    result = ""
    for header, starts, ends in headers_pos:
        result += input_str[starts:ends]
    return result
