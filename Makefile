
download:
	while IFS= read -r line; do \
		bsub -M 25000 -R "rusage[mem=20000]" 'python src/python/download_tissue.py -t \"$$line\"'; \
		sh src/bash/download_tissues.sh; \
	done < textfiles/ID_tissue_list.txt;


sample_patches:
		while IFS= read -r pair; \
			do bsub -o "log/$${pair}.out" -e "log/$${pair}.err" -M 5000 -R 'rusage[mem=5000]' "python src/python/sample_tissue_patches.py -p '$${pair}'"; \
   	done < textfiles/ID_tissue_list.txt
#bsub -Is -M 10000 -R 'rusage[mem=10000]' "python src/python/sample_tissue_patches.py -p 'GTEX-R55G-0826 Lung'"

generate_lung_retrained_inception_representations:
	for ID in `cat textfiles/lung_IDs.txt`; \
		do bsub -M 100000 -R 'rusage[mem=100000]' -P gpu "python src/python/generate_features.py -i $$ID -m retrained"; \
   	done; \

# bsub -Is -M 100000 -R 'rusage[mem=100000]' -P gpu "python src/python/generate_finetuned_inceptionet_lung_features.py -i GTEX-13VXU-2726"
generate_lung_raw_inception_representations:
	for ID in `cat textfiles/lung_IDs.txt`; \
		do bsub -M 100000 -R 'rusage[mem=100000]' -P gpu "python src/python/generate_raw_inceptionet_lung_features.py -i $$ID -m raw"; \
   	done; \
# bsub -Is -M 100000 -R 'rusage[mem=100000]' -P gpu "python src/python/generate_raw_inceptionet_lung_features.py -i GTEX-13VXU-2726"
