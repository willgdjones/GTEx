graphs:
	rsync -vr --delete willj@ebi-cli-002.ebi.ac.uk:/hps/nobackup/research/stegle/users/willj/GTEx/figures /Users/fonz/Documents/Projects/GTEx

rsync:
	rsync -vr --delete --exclude '*_pycache_*' /Users/fonz/Documents/Projects/GTEx/src willj@ebi-cli-002.ebi.ac.uk:/hps/nobackup/research/stegle/users/willj/GTEx
	rsync -vr /Users/fonz/Documents/Projects/GTEx/Makefile willj@ebi-cli-002.ebi.ac.uk:/hps/nobackup/research/stegle/users/willj/GTEx
	rsync -vr /Users/fonz/Documents/Projects/GTEx/README.md willj@ebi-cli-002.ebi.ac.uk:/hps/nobackup/research/stegle/users/willj/GTEx
	rsync -vr --exclude '*_pycache_*' /Users/fonz/Documents/Projects/GTEx/tests willj@ebi-cli-002.ebi.ac.uk:/hps/nobackup/research/stegle/users/willj/GTEx

results:
	rsync -vr --delete willj@ebi-cli-002.ebi.ac.uk:/hps/nobackup/research/stegle/users/willj/GTEx/results /Users/fonz/Documents/Projects/GTEx


monitor:
	python src/python/monitor.py

download:
	while IFS= read -r line; do \
		bsub -M 25000 -R "rusage[mem=20000]" 'python src/preprocessing/download_tissue.py -t \"$$line\"'; \
		sh src/bash/download_tissues.sh; \
	done < textfiles/ID_tissue_list.txt;


sample_patches:
	while IFS= read -r pair; \
		do bsub -o "log/$${pair}.out" -e "log/$${pair}.err" -M 10000 -R 'rusage[mem=10000]' "python src/preprocessing/sample_patches.py -p '$${pair}'"; \
		 done < textfiles/ID_tissue_list.txt
#bsub -Is -M 10000 -R 'rusage[mem=10000]' "python src/python/sample_patches.py -p 'GTEX-R55G-0826 Lung'"

generate_features:
	while IFS= read -r pair; \
		do bsub -o "log/$${pair}.out" -e "log/$${pair}.err" -M 110000 -R 'rusage[mem=110000]' -P gpu "python src/features/generate_features.py -p '$${pair}'"; \
		 done < textfiles/ID_tissue_list.txt


overnight:
	bsub -Is -M 80000 -R "rusage[mem=80000]" 'python src/features/collect_features.py'
	bsub -Is -M 80000 -R "rusage[mem=80000]" 'python src/features/aggregate_features.py'

expression_associations:
	while IFS= read -r key; \
		do bsub -o "log/expression_associations_$${key}.out" -e "log/expression_associations_$${key}.err" -M 10000 -R "rusage[mem=10000]" "python src/investigations/TFCorrectedFeatureAssociations.py -g TFCorrectedFeatureAssociations -n compute_pvalues -p '$${key}'"; \
		 done < textfiles/parameter_combinations.txt
