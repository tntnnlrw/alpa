#!/bin/bash --login
start_time=$(date +%s)

# gpus=(4 8)
# bs=(32 64 128 256 512 1024)
gpus=(4 8)
bs=(32 64 128 256 512 1024)

for ((k=0; k<${#gpus[*]}; k=k+1)); do
    for ((j=0; j<${#bs[*]}; j=j+1)); do
        # python read_dump.py --load_file=dump_${gpus[k]}gpu_${bs[j]}bs.txt --random --memory
        python read_dump.py --load_file=dump_${gpus[k]}gpu_${bs[j]}bs.txt
        python read_dump.py --load_file=dump_${gpus[k]}gpu_${bs[j]}bs.txt --random
        # python read_dump.py --load_file=dump_${gpus[k]}gpu_${bs[j]}bs.txt --memory
    done
done




end_time=$(date +%s)
cost_time=$[ $end_time - $start_time]
echo "run.sh spends $(($cost_time/60))min $(($cost_time%60))s"
