#!/bin/bash


configFile=${1:-"final-ag350-fop85-expe36900-32runs.yaml"}
base_fop=${2:-"30"}
pattern_ag=${3:-"350"}
pattern_fop=${4:-"85"}
singularity_image_filename=${5:-"spectral-kilo.simg"}
ag_min=${6:-"10"}
ag_step=${7:-"5"}
ag_max=${8:-"80"}

results_dir="results"
conf_dir="conf"
runKilombo_filename="./scripts/runKilombo.py"
ag_lst=$(seq $ag_min $ag_step $ag_max)

base_config=$(echo $configFile | sed "s/ag$pattern_ag/agAG/" | sed "s/fop$pattern_fop/fop$base_fop/" | sed "s/$conf_dir\///" | sed 's/.yaml//')
tmp_base_dir=/tmp/limmsswarm/$base_config
tmp_config_dir=$tmp_base_dir/config
tmp_results_dir=$tmp_base_dir/results

# Create temporary directories
mkdir -p $tmp_base_dir
mkdir -p $tmp_config_dir
mkdir -p $tmp_results_dir

# Handle each configuration case
for ag in $(echo $ag_lst); do
    # Generate config files
    case_name=$(echo $base_config | sed "s/agAG/ag$ag/")
    case_config_path=$tmp_config_dir/$case_name.yaml
    echo Processing \'$case_name\'...
    cat $conf_dir/$configFile | sed "s/commsRadius: $pattern_fop/commsRadius: $base_fop/" | sed "s/nBots: $pattern_ag/nBots: $ag/" > $case_config_path

    # Launch runKilombo script on each config
    singularity exec $singularity_image_filename $runKilombo_filename -c $case_config_path -o $tmp_results_dir

    # Copy back results, and then remove them when finished
    echo Copying back results into \'$results_dir\'
    { rsync -av $tmp_results_dir/$case_name $results_dir/; rm -fr $tmp_results_dir/$case_name; } &
done

# Copy back results
echo Finished all simulations.
wait

# Delete temp data
rm -fr $tmp_results_dir
rm -fr $tmp_config_dir
rm -fr $tmp_base_dir

