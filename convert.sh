#!/bin/bash
gcc -o "dumpdbd.out" ./dump_dbd.cpp 

for file in ./original/*.DBD; do
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    now=$(date +'%Y-%m-%d %H:%M:%S')

    echo "$now - INFO    :./dumped/$filename.csv and ./dumped_head/$filename.csv was created for location=$1" >> ./lysi_data.log
    ./dumpdbd.out -n -i $file > "./dumped/$filename.csv"
    ./dumpdbd.out -d -i $file > "./dumped_head/$filename.csv"
    python preprocess.py ./dumped/"$filename".csv $1
    python postprocess.py ./excel/"$filename".xlsx $1
done
