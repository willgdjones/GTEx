while IFS= read -r line; do
    bsub -Is -M 100000 -R "rusage[mem=100000]" -P gpu "python src/python/generate_covering_representations.py -t \"$line\" -m models/inception_50_-1.h5"
done < textfiles/ID_tissue_list.txt


