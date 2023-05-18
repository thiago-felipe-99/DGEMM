#!/usr/bin/env bash

if [[ ! -f "out/result.txt" ]]; then
  out/dgemm -l 1507 -s -d simple > out/simple.txt
fi

out/dgemm -l 1507 -s -d $1 > out/result.txt && diff out/simple.txt out/result.txt
