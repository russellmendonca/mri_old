

from shutil import copyfile
soureDir = '/home/russellm/data/s3/maml-gps/doodad/logs/Sawyer-Push-3D-block-v1/trpoExperts_20X20/'
targetDir = '/home/russellm/mri_onPolicy/expertPolicyWeights/TRPO-push-20X20-v1/'
selItr = 300
numTasks = 20

for task in range(numTasks):
	src = soureDir+'Task_'+str(task)+'/itr_'+str(selItr)+'.pkl'
	dest = targetDir + 'Task'+str(task)+'/itr_'+str(selItr)+'.pkl'

	copyfile(src, dest)

