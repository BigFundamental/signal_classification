#! /bin/bash

thread_num=8
dir=''
i=0
for((i=0;i<thread_num;++i));
do
    python ./tools/runPredict.py -d '/Users/changkong/ML/Signal Classification/testData/Data.20180512/' --output='./test.result' -f 'Channel_1.csv' --enable_timing --server=http &> perf.$i &
done

wait

exit 0
