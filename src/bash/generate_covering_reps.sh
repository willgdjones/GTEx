for ID in `cat $1`; do
    echo $ID
    bsub -M 50000 -R "rusage[mem=50000]" -P gpu "python src/generate_covering_representations.py -i $ID"
done
