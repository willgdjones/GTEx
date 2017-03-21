#while IFS= read -r line; do
    #bsub -Is -M 50000 -R "rusage[mem=50000]" "python src/python/aggregate_reps_and_form_expression_X_y.py -t \"$line\" -m inception_50_-1.h5 -a mean"
#done < textfiles/tissue_types.sh

bsub -Is -M 50000 -R "rusage[mem=50000]" "python src/python/aggregate_reps_and_form_expression_X_y.py -t 'Liver' -m inception_50_-1.h5 -a mean"
