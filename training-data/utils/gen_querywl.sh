QUERYWORDS_PATH='./training-data/utils/querywords'
QUERY_FILE='./training-data/utils/query.wl'

DATA=$(find ${QUERYWORDS_PATH} -name '*.wl')

for f in $DATA; do
    sed -i '' -e '$a\' $f
done

cat ${DATA} | sort -u > ${QUERY_FILE}