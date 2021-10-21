import datetime
import time
import uuid

from fabric import task


def timestamp():
    return str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')


PATH_PROJECT = '~/hpc_tutorial/files'  # project folder on the hpc cluster


def create_pbs(settings, script_arguments, pbs_arguments=[]):
    '''Creates a pbs string.'''

    script_arguments_str = ' '.join(['--%s %s' % (key, val) for key, val in script_arguments.items()])

    pbs_arguments_str = '\n#\n'.join([f'#PBS {arg}' for arg in pbs_arguments])

    job_string = '''#!/bin/bash -l
#
#PBS -l nodes=1:ppn=4:anygtx,walltime=24:00:00
#
# job name
#PBS -N {job_name}
#
{pbs_arguments}

# load cuda module first
module load cuda/10.2

# jobs always start in $HOME
mkdir -p {path_project}
cd {path_project}

# activate virtual environment
source ~/miniconda3/bin/activate {env_name}

# run script
python {script_name} {script_arguments}

'''.format(**settings, script_arguments=script_arguments_str, pbs_arguments=pbs_arguments_str)

    return job_string


def run_pbs(c, pbs):
    with c.cd(PATH_PROJECT):
        # treat pbs string as file stream via the here-document syntax (https://en.wikipedia.org/wiki/Here_document)
        result = c.run('PATH=$PATH:/apps/torque/current/bin && qsub.tinygpu << EOF\n%s\nEOF' % pbs, shell=True)
    return result.stdout.strip()


@task
def experiment(c):
    '''Runs experiment in nested loop'''

    for epochs in [1, 2, 4, 8]:
        for batch_size in [4, 16, 64, 256]:

            key = f'e{epochs}-b{batch_size}'
            cur_job_name = 'exp_%s_%s_%s' % (key, timestamp(), uuid.uuid4().hex[:6])

            cur_settings = {
                'script_name': 'mnist_minimal.py',
                'path_project': PATH_PROJECT,
                'env_name': 'dl_test',
                'job_name': cur_job_name}

            cur_script_arguments = {
                'epochs': epochs,
                'batchsize': batch_size,
                'outfile': 'results/result_' + key + '.json',
                }

            cur_pbs = create_pbs(cur_settings, cur_script_arguments)
            run_pbs(c, cur_pbs)
            time.sleep(0.2)


@task
def experiment2(c):
    '''Runs multiple jobs as a sequence (each job depends on previous one)'''

    epochs = 10
    batch_size = 64
    previous_job_id = None

    for i in range(3):
        key = f'e{epochs}-b{batch_size}-i{i}'
        cur_job_name = 'exp_%s_%s_%s' % (key, timestamp(), uuid.uuid4().hex[:6])

        cur_settings = {
            'script_name': 'mnist_minimal.py',
            'path_project': PATH_PROJECT,
            'env_name': 'dl_test',
            'job_name': cur_job_name}

        cur_script_arguments = {
            'epochs': epochs,
            'batchsize': batch_size,
            'outfile': f'results/result_{key}.json',
            'save_weights': f'weights/{i}.h5'
            }

        if i == 0:
            pbs_arguments = []
        else:
            pbs_arguments = [f'-W depend=afterok:{previous_job_id}']
            cur_script_arguments['load_weights'] = f'weights/{i-1}.h5'

        cur_pbs = create_pbs(cur_settings, cur_script_arguments, pbs_arguments)
        previous_job_id = run_pbs(c, cur_pbs)
        time.sleep(0.2)
