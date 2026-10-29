"""Deterministic train/validation/test assignment by claim_id.

CRC-32 of the claim_id gives the same bucket in Python (zlib.crc32) and in Spark
(F.crc32), so notebooks, jobs and tests all agree on which claims are held out
without storing a split table.
"""
from __future__ import annotations

import zlib

SPLITS = (("train", 70), ("val", 85), ("test", 100))   # cumulative percent


def split_for(claim_id: str) -> str:
    bucket = zlib.crc32(claim_id.encode("utf-8")) % 100
    for name, upper in SPLITS:
        if bucket < upper:
            return name
    return "test"


def spark_split_col(col_name: str = "claim_id"):
    """Same assignment as split_for, as a Spark Column expression."""
    from pyspark.sql import functions as F
    bucket = F.crc32(F.col(col_name)) % 100
    return (F.when(bucket < 70, "train").when(bucket < 85, "val").otherwise("test"))
