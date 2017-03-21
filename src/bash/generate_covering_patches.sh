#for ID in `cat $1`; do
    #echo $ID
    #bsub -M 10000 -R "rusage[mem=10000]" "python src/generate_covering_lung_patches.py -i $ID"
#done
while IFS= read -r line; do
   bsub -M 10000 -o 'log/tissue_patches.out' -e 'log/tissue_patches.err' -R "rusage[mem=10000]" "python src/python/generate_covering_tissue_patches.py -i '$line'"
   #bsub -Is -M 10000 -R "rusage[mem=10000]" "python src/python/generate_covering_tissue_patches.py -i '$line'"
done < $1 
