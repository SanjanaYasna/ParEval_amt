#OVERALL COMMANDS:

0. python parse_to_txt.py /path/to/json to get the raw txt of generated outputs for inspection 

1. python clean_output_bulk.py -i /path/to/dir/of/jsons 
-> /path/to/dir/of/jsons/cleaned containing files as input for driver runs 
OR SPECIFIC CLEANING SCRIPTS ie futures_promises/cache/clean.py and locking_contention/cache/bulk_clean.py 

2. Run step 0 again to validate cleaned output 

3. RUN 1.5.1 and 1.10.0 and get results jsons

4. python combine_runs.py /path/to/prompt/first/level/dir ie futures_promises 
-> get a path under /drivers/tcmalloc of the combined results 

5. python create-dataframe-bulk.py /path/to/dir/of/jsons 
-> /path/to/dir/of/jsons with a bunch of .csv files of json equivalents 

6. python metrics_from_dir.py /path/to/dir/of/jsons -n 100  --problem-size /work/pi_mrobson_smith_edu/ParEval_amt/drivers/problem-sizes.json
-> /path/to/dir/of/jsons/data.csv
Interchangeable with metrics_problem.py 

7. python runtimes.py /path/to/dir/of/csvs
python specific_runtimes.py /path/to/dir/of/jsons 



Redoing things template: 

cd transform/driver/tcmalloc
rm *.csv
cd -
python combine_runs.py transform/
python create-dataframe_bulk.py transform/driver/tcmalloc/
python metrics_from_dir.py transform/driver/tcmalloc/ -n 100  --problem-size /work/pi_mrobson_smith_edu/ParEval_amt/drivers/problem-sizes.json
python runtimes.py transform/driver/tcmalloc/
python specific_runtimes.py transform/driver/tcmalloc/




python ../replace_entry.py magicoder_combined.json /work/pi_mrobson_smith_edu/scratch/generation_hpx/to_merge/pass1_combined/09_magicoder_combined.json magicoder_2.json
