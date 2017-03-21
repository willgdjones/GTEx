download:
	sh src/bash/download_tissues.sh

covering_patches:
	sh src/bash/generate_covering_patches.sh textfiles/ID_tissue_list.txt 

reps:
	sh src/bash/generate_covering_reps.sh

aggregations:
	sh src/bash/aggregate_reps_and_form_expression_X_y.sh



