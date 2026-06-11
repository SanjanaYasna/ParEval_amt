# Analysis Scripts to Run

## @k.py : 

    get @k results in the following folder organization:
    @k/
        overall_summary.csv
        per_prompt_summary.csv
    MAKE SURE BEFOREHAND: adjust the paths in CSV_PATHS to reflect where your data.csv files are  

## generation_metrics.py:
    get generation metrics per model (think memory, gpu utilization, etc) in following organization
    visuals_specific/
        model_metrics_by_category.csv
        model_metrics_overall.csv
    MAKE SURE BEFOREHAND: adjust the paths in CSV_PATHS to reflect where your json_runtime_summary.csv files are  

## graphs.py 
    get historgrams of speedup, efficiency, and strict_pass @1
    Outputs like this:
    visuals/
        speedup1.png
        strict_pass1.png
        efficiency1.png
    MAKE SURE BEFOREHAND: comment in/out the METRICS dictionary items for which graphs you want 
    FOR HORIZONTAL HISTOGRAMS (which I think is the better layout), just run  `sideways_histograms.py` instead and make sure to adjust the METRICS dict as needed too. It won't overwrite the original histogram graphs as they will have _horizontal in the name

## graph_threads.py
    get runtime graphs on a per-prompt basis among all the models (this is regardless of strict_pass, so long as the code just passes its runtime counts) 
    output is like:
    visuals_specific/
        fft/ ...
        futures_promises/ ...
        ...

## graph_threads_strict_pass_only.py
    get runtime graphs among runtimes that specifically pass strict_pass, it also adds the counts next to the lines, unlike graph_threads
    output is like:
    visuals_specific_strict_pass/
        fft/ ...
        futures_promises/ ...
        ...
