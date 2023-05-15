#!/usr/bin/env bash
#
for i in $(eval echo {$2..$3..$4})
do
  $1 $i
done
