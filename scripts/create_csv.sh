#!/usr/bin/env bash

for i in {1..5}; do
	echo Rounding $i
	for file in $1; do
		filename=$(echo "$file" | sed "s|\(.*\)\/\(.*\)$|\2|")
		out="./out_csv/raw/$filename.$3.$i.csv"
		echo Making csv for $filename
		if [ "$4" = "true" ]; then
      $file -d $5 -o $2 -p >$out
		else
      $file -d $5 -o $2 >$out
		fi
	done
done
