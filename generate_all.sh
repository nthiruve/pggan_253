#!/bin/bash
FILES=test_images/*
for f in $FILES ; do (
f_proc=${f:12:-4} &&
echo "$f_proc" && 
caption_path=`find flowers/ -name $f_proc.txt` &&
echo "$caption_path" &&
python generate_images.py $f_proc $caption_path -1); done
