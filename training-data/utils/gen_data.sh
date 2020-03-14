cd training-data/utils
python3 replaceunknoenwords.py ../$1/$2/$1.txt
python3 txt2ctf.py --map query.wl intent.wl slots.wl --annotated True --input ../$1/$2/$1.txt --output ../$1/$2/$1.ctf