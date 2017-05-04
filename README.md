# Computational Genomics Final Project #

In this project, we are proposing a classification engine using single-nucleotide polymorphisms (SNPs) to predict patients who are significantly at risk of developing HGPCa. 

For a detailed understanding of the study, see the proposal attached to this repo.

## Things To Do ##
- [X] Setup data pipeline
- [X] Parse gleason labels
- [X] Tie gleason labels to entity IDs in .maf file
- [X] Parse SNP data
- [X] Tie SNP data to gleason labels
- [X] Implement Cross Validation
- [X] Evaluate performance of different methods 
- [X] Write final paper
- [X] Write powerpoint

## Running Instructions ##
data_directory='/your/data/directory'
results_directory='/your/results/directory'
sh process_data.sh $data_directory $results_directory
sh train_model.sh $data_directory $results_directory
sh test_model.sh $data_directory $results_directory
