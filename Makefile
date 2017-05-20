
download:
	while IFS= read -r line; do \
		bsub -M 25000 -R "rusage[mem=20000]" 'python src/python/download_tissue.py -t \"$$line\"'; \
		sh src/bash/download_tissues.sh; \
	done < textfiles/ID_tissue_list.txt;


sample_patches:
	while IFS= read -r pair; \
		do bsub -o "log/$${pair}.out" -e "log/$${pair}.err" -M 10000 -R 'rusage[mem=10000]' "python src/python/sample_patches.py -p '$${pair}'"; \
   	done < textfiles/ID_tissue_list.txt
#bsub -Is -M 10000 -R 'rusage[mem=10000]' "python src/python/sample_patches.py -p 'GTEX-R55G-0826 Lung'"

generate_features:
	while IFS= read -r pair; \
		do bsub -o "log/$${pair}.out" -e "log/$${pair}.err" -M 100000 -R 'rusage[mem=100000]' -P gpu "python src/python/generate_features.py -p '$${pair}'"; \
   	done < textfiles/ID_tissue_list.txt
#bsub -Is -M 100000 -R 'rusage[mem=100000]' "python src/python/generate_features.py -p 'GTEX-R55G-0826 Lung'"

