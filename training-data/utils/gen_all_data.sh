DATA=$(find ./training-data -name '*.txt')

cd training-data/utils

while read -r line; do
    FILE_NAME=$(echo "../.$line") 
    FILE_NAME_NO_EXT=${FILE_NAME%.*} 
    CTF_FILE_NAME=$(echo "$FILE_NAME_NO_EXT.ctf") 

    python3 replaceunknoenwords.py $FILE_NAME
    python3 txt2ctf.py --map query.wl intent.wl slots.wl --annotated True --input $FILE_NAME --output $CTF_FILE_NAME
done <<< "$DATA"
