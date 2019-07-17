cd training-data/utils
python replaceunknoenwords.py ../$1/$2/0.$1.txt
python txt2ctf.py --map query.wl intent.wl slots.wl --annotated True --input ../$1/$2/0.$1.txt --output ../$1/$2/0.$1.ctf