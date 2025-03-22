#########################################################################
# File Name: generate_commands.sh
# Author: longxing
# mail: longxing@uw.edu
# Created Time: Mon 17 Feb 2020 05:03:43 PM PST
#########################################################################
#!/bin/bash

rosetta="rosetta_scripts"
source_folder="./prepare_inputs/"
cur_folder=$(pwd)
iterations=200

for ii in *bp;do
    tag=$(echo ${ii%.bp})
    mkdir ${tag}
    mv $ii ${tag}
    cp ${source_folder}/design_2019CoV.xml ${tag}
    cp ${source_folder}/input.flags ${tag}
    cp ${source_folder}/motif.pdb ${tag}
    cp ${source_folder}/target.pdb ${tag}
    cp ${source_folder}/design_2019CoV.xml ${tag}
    echo "cd ${cur_folder}/${tag}; ${rosetta}  @input.flags -parser:protocol design_2019CoV.xml -parser:script_vars blueprint=${tag}.bp dumpname=${tag} iterations=${iterations}"
done > commands.list

cp ${source_folder}/array_jobs.sh .

echo "commands generation done. Run the following command to submit the jobs:"
echo "######################################################################"
echo ""
echo "sbatch -a 1-$(cat commands.list|wc -l) array_jobs.sh"
echo ""
echo "######################################################################"
