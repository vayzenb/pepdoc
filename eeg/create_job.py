"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/pepdoc' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)

import subprocess
from glob import glob
import os
import time
import pdb

mem = 16
run_time = "3-00:00:00"

#subj info
#stim info
study_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn/model_training'
stim_dir = f'/lab_data/behrmannlab/image_sets/'

script_list = ['decode_category','extract_erp','pairwise_analysis']




def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
# Submit job to cpu queue                
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
# Job memory request
#SBATCH --mem={mem}gb
# Time limit days-hrs:min:sec
#SBATCH --time {run_time}
# Exclude
#SBATCH --exclude=mind-1-26,mind-1-30
# Standard output and error log
#SBATCH --output={curr_dir}/slurm_out/{job_name}.out

conda activate ml_new

echo "images transferred"
python {script_name}
"""
    return sbatch_setup



# run low-demand scripts
for script in script_list:
    job_name = script
    script_path = f'{curr_dir}/eeg/{script}.py'
    print(job_name)

    #create sbatch script
    f = open(f"{job_name}.sh", "a")
    f.writelines(setup_sbatch(job_name, script_path))
    
    f.close()
    
    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")

rois = ['occipital','frontal','dorsal','ventral']
#run high-demand scripts
for roi in rois:
    job_name = f'{script}_{roi}'
    script_path = f'{curr_dir}/eeg/{script}.py {roi}'
    print(job_name)

    #create sbatch script
    f = open(f"{job_name}.sh", "a")
    f.writelines(setup_sbatch(job_name, script_path))
    
    f.close()
    
    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")

