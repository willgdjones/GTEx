#clean-reps:
	#rm -rf data/processed/assembled_representations/inception_50_-1.h5/*

#clean-covering-patches:
	#rm -rf data/processed/covering_patches/$(size)/*

download:
	while IFS= read -r line; do \
		bsub -M 25000 -R "rusage[mem=20000]" 'python src/python/download_tissue.py -t \"$$line\"'; \
		sh src/bash/download_tissues.sh; \
	done < textfiles/ID_tissue_list.txt;

all-covering-patches:
	while IFS= read -r line; do \
	   bsub -M 10000 -o 'log/tissue_patches.out' -e 'log/tissue_patches.err' -R "rusage[mem=10000]" "python src/python/generate_covering_tissue_patches.py -i \"$$line\" -s $(size)"; \
	done < textfiles/ID_tissue_list.txt;

test-covering-patches:
	   bsub -Is -M 10000 -R "rusage[mem=10000]" "python src/python/generate_covering_tissue_patches.py -i 'GTEX-XUJ4-1426 Lung' -s $(size) -r '1'"; \


all-reps:
	while IFS= read -r line; do \
		bsub -M 100000 -R "rusage[mem=100000]" -P gpu "python src/python/generate_covering_representations.py -t \"$$line\" -m inception_50_-1.h5 -s $(size)"; \
	done < textfiles/tissue_types.txt;

reps:
	bsub -Is -M 100000 -R "rusage[mem=100000]" -P gpu -q production-rh7	"python src/python/generate_covering_representations.py -t 'Lung' -m inception_50_-1.h5 -s $(size) -r $(regenerate)"

all-exp-aggregations:
	while IFS= read -r tissue; do \
		bsub -M 50000 -R "rusage[mem=50000]" -P gpu "python src/python/aggregate_reps_and_form_expression_X_y.py -t \"$$tissue\" -m inception_50_-1.h5 -a $(aggregation) -s $(size)"; \
	done < textfiles/tissue_types.txt; \

exp-aggregations:
	bsub -Is -M 50000 -R "rusage[mem=50000]" "python src/python/aggregate_reps_and_form_expression_X_y.py -t $(tissue) -m inception_50_-1.h5 -a $(aggregation) -s $(size)"

gen-aggregations:
	bsub -M 50000 -R "rusage[mem=50000]" "python src/python/aggregate_reps_and_form_genotype_X_y.py -t $(tissue) -m inception_50_-1.h5 -a $(aggregation) -s $(size)"

all-tissue-associations:
	#Very long. Run overnight
	while IFS= read -r tissue; do \
		for i in `seq 0 1023`; do \
			bsub -M 10000 -o "log/${i}.out" -e "log/${i}.err" -R 'rusage[mem=10000]' "python src/python/run_tissue_expression_association_tests.py -f 0 -c $$i -u 1 -m inception_50_-1.h5 -t \"$$tissue\" -a $(aggregation) -s $(size)"; \
			bsub -M 10000 -o "log/${i}.out" -e "log/${i}.err" -R 'rusage[mem=10000]' "python src/python/run_tissue_expression_association_tests.py -f 1 -c $$i -u 1 -m inception_50_-1.h5 -t \"$$tissue\" -a $(aggregation) -s $(size)"; \
		done; \
	done < textfiles/tissue_types.txt;


single-tissue-association:
	for i in `seq 0 1023`; do \
		bsub -M 5000 -o "log/${i}.out" -e "log/${i}.err" -R 'rusage[mem=5000]' "python src/python/run_tissue_expression_association_tests.py -f 0 -c $$i -u 0 -m inception_50_-1.h5 -t $(tissue) -a $(aggregation) -s $(size)"; \
		bsub -M 5000 -o "log/${i}.out" -e "log/${i}.err" -R 'rusage[mem=5000]' "python src/python/run_tissue_expression_association_tests.py -f 1 -c $$i -u 0 -m inception_50_-1.h5 -t $(tissue) -a $(aggregation) -s $(size)"; \
	done;

single-tissue-component-association:
	bsub -Is -M 5000 -R 'rusage[mem=5000]' "python src/python/run_tissue_expression_association_tests.py -f 0 -c $(component) -u $(upper_limit) -m inception_50_-1.h5 -t $(tissue) -a $(aggregation) -s $(size)"; \
	bsub -Is -M 5000 -R 'rusage[mem=5000]' "python src/python/run_tissue_expression_association_tests.py -f 1 -c $(component) -u $(upper_limit) -m inception_50_-1.h5 -t $(tissue) -a $(aggregation) -s $(size)"; \

all-graphs:
	while IFS= read -r tissue; do \
		bsub -M 20000 -R 'rusage[mem=20000]' "python src/python/generate_expression_association_graphs.py -t \"$$tissue\" -m inception_50_-1.h5"; \
	done < textfiles/tissue_types.txt;

monitor:
	python src/python/monitor.py
