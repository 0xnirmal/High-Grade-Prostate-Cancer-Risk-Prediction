# Computational Genomics Final Project #

In this project, we are proposing a classification engine using somatic single-nucleotide polymorphisms (SNPs) to predict patients who are significantly at risk of developing HGPCa. 

For a detailed understanding of the study, see the writeup attached to this repo.

## Running Instructions ##
```
data_directory='/data'
results_directory='/results'
sh process_data.sh $data_directory $results_directory
sh train_model.sh $data_directory $results_directory
sh test_model.sh $data_directory $results_directory
```
## Things To Do ##
- [X] Setup data pipeline
- [X] Parse gleason labels
- [X] Tie gleason labels to entity IDs in .maf file
- [X] Parse SNP data
- [X] Tie SNP data to gleason labels
- [X] Implement cross validation
- [X] Use Sci-kit for 3 models and evaluate on cross-validation data
- [X] Evaluate performance of different methods 
- [X] Write final paper
- [X] Write powerpoint

