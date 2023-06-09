#!/usr/bin/env python
import os
import sys

for __fh in os.listdir(sys.argv[1]):
    # Fix separator issues
    fh = __fh.replace("=", "-")
    fh = fh.replace("128walk", "128_walk")

    # Rearrange embedding attribute in names
    _, fh = fh.split("cache_system-LFR_")
    _fh, fh_ = fh.split("embedding-")
    method, *postface = fh_.split("_")
    new_name = f"cache_system-LFR_embedding-{method}_{_fh}_{'_'.join(postface)}"

    # Correct introduced repeated separators
    new_name = new_name.replace("__", "_")

    # Remove unneeded information
    new_name = new_name.replace("_layers-1-2", "")

    # Rename file with new name
    os.rename(f"{sys.argv[1]}/{__fh}", f"{sys.argv[1]}/{new_name}")
