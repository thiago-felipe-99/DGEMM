#!/usr/bin/env bash

 for i in {1..2}; do
 	echo Rounding $i
 	for file in $1; do
    filename=$(echo "$file" | sed "s|\(.*\)\/\(.*\)$|\2|")
 		out="./out_csv/raw/$filename.$4.$i.csv"
 		echo Making csv for $filename
    $file -d $2 -p $3 >$out
 	done
 done
