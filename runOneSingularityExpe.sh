#!/bin/bash

configFile=${1:-"final-ag350-fop85-expe36900-32runs.yaml"}
singularity_image_filename=${2:-"spectral-kilo.simg"}

results_dir="results"
conf_dir="conf"
runKilombo_filename="./scripts/runKilombo.py"

base_config=$(echo $configFile | sed "s/$conf_dir\///" | sed 's/.yaml//')
tmp_base_dir=/tmp/limmsswarm/$base_config

# Create temporary directories
mkdir -p $tmp_base_dir

# Launch runKilombo script on each config
singularity exec $singularity_image_filename $runKilombo_filename -c $conf_dir/$configFile -o $tmp_base_dir

# Copy back results, and then remove them when finished
echo Copying back results into \'$results_dir\'
{ rsync -av $tmp_base_dir/$base_config $results_dir/; rm -fr $tmp_base_dir/$base_config; }

# Delete temp data
rm -fr $tmp_base_dir

