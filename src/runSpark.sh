#!/bin/bash
START=$(date +%s)

export deepbind_env='local'
#export deepbind_env='m2'

#python -m molgen.TrainMolGenModel 


#python -m controller.SimulateAssays  --runmode test

python -m controller.SimulateAssays  --runmode genFromImplicit

#python -m utils.TestPipeline


END=$(date +%s)


DIFF=$(( $END - $START ))

echo "It took $DIFF seconds"
