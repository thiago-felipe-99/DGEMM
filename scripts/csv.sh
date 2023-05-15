#!/usr/bin/env bash

for i in {1..10}; do
  echo Rounding $i
	for file in "$1"/*; do
		filename=$(echo "$file" | sed "s|$1/||")
		out="./out_csv/raw/$filename.$i.csv"
		echo Making csv for $filename
		./exec_stress.sh "$1/$filename" 1 1028 1 >$out
	done
done
for file in "$1"/*; do
	filename=$(echo "$file" | sed "s|$1/||")
  awk -F , '{a[FNR]+=$2;b[FNR]+=$3;c[FNR]++;}END{for(i=1;i<=FNR;i++)print $1,a[i]/c[i],b[i]/c[i];}' ./out_csv/raw/$filename.* > "./out_csv/$filename.csv"
done
