#!/bin/bash
gcc -o "dumpdbd.out" ./dump_dbd.cpp 

filename=$(basename -- "$2")
extension="${filename##*.}"
filename="${filename%.*}"
now=$(date +'%Y-%m-%d %H:%M:%S')

echo $filename
echo $1
echo $2

echo "$now - INFO    :./dumped/$filename.csv and ./dumped_head/$filename.csv was created for location=$1" >> ./lysi_data.log
./dumpdbd.out -n "./original/$2" > "./dumped/$filename.csv"
./dumpdbd.out -d "./original/$2" > "./dumped_head/$filename.csv"
python preprocess.py ./dumped/"$filename".csv $1
python postprocess.py ./excel/"$filename".xlsx $1
