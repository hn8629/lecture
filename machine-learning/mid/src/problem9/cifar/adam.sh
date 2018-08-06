
#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=05:00:00
#$ -o results/$JOB_ID.out
#$ -e results/$JOB_ID.err
#$ -N cifar10-example

. /etc/profile.d/modules.sh
module load cuda
module load nccl

#----- pyenv
export PYENV_ROOT=$HOME/.pyenv
if [ -d "${PYENV_ROOT}" ]; then
   export PATH=${PYENV_ROOT}/bin:$PATH
   eval "$(pyenv init -)"
fi


pyenv local 3.6.5
#python -c "import sys; print(':'.join(x for x in sys.path if x))"

python train_cifar.py \
-opt adam \
-o cifar_result/adam-default \
--epoch 200
