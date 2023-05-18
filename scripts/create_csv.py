#!/usr/bin/env python

import subprocess


def create_command(name="dgemm", unroll=8, block_size=32):
    return ["gcc", "-O3", "-fopenmp", "-march=native", "src/main.c",
            "src/dgemm.c", "-o", "./out/"+name, "-DUNROLL="+str(unroll),
            "-DBLOCK_SIZE="+str(block_size), "-lm"]


min_unroll = 1
max_unroll = 8

subprocess.run(create_command(), check=True)

print("Binary built successfully!")
