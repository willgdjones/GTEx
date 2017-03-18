
for i in `seq 0 1023`; do
    bsub -M 5000 -o "log/${i}.out" -e "log/${i}.err" -R 'rusage[mem=5000]' "python src/run_association_tests.py -f $1 -s 0 -c $i -u 0"
    bsub -M 5000 -o "log/${i}.out" -e "log/${i}.err" -R 'rusage[mem=5000]' "python src/run_association_tests.py -f $1 -s 1 -c $i -u 0"
done
