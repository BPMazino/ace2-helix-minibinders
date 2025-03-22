#########################################################################
# File Name: run.sh
# Author: longxing
# mail: longxing@uw.edu
# Created Time: Mon 17 Feb 2020 04:33:18 PM PST
#########################################################################
#!/bin/bash

cd /net/scratch/longxing/Scaffolds/2019nCoV_test_run/22HBAAB0HM0HBAAB22H; /net/scratch/longxing/Scaffolds/Rosetta_scaffolds/source/cmake/build_release/rosetta_scripts  @input.flags -parser:protocol design_2019CoV.xml -parser:script_vars blueprint=22HBAAB0HM0HBAAB22H.bp dumpname=22HBAAB0HM0HBAAB22H iterations=500
