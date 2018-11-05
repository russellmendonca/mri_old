import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
import pickle
print('a')
# sourcePrefix = "/home/russellm/data/s3/Sawyer-Push-3D-block/trpoExperts_20X20/Task_"
# targetPrefix = "/home/russellm/mri_onPolicy/expertPolicyWeights/TRPO-push-20X20/"
#sourcePrefix = "/home/russellm/data/s3/Sawyer-Push-3D-block/trpoExperts_20X20/Task_"
#targetPrefix = "/home/russellm/mri_onPolicy/expertPolicyWeights/TRPO-push-20X20/"
sourcePrefix = '/home/russellm/data/s3/maml-gps/doodad/logs/Sawyer-Push-3D-block-v1/trpoExperts_20X20/Task_'
targetPrefix = '/home/russellm/mri_onPolicy/expertPolicyWeights/TRPO-push-20X20-v1/'


import os
if os.path.isdir(targetPrefix)!=True:
    os.mkdir(targetPrefix)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True


for fileItr in [300]:
    
    print('fileItr'+str(fileItr))
    for task in range(20):

        tf.reset_default_graph()   
        with tf.Session(config = config) as sess:
            sourceFile = sourcePrefix+str(task)+'/itr_'+str(fileItr)+'.pkl'
            targetFolder = targetPrefix + "Task_"+str(task)
            if os.path.isdir(targetFolder)!=True:
                os.mkdir(targetFolder)
            targetFile = targetFolder + '/itr_'+str(fileItr)+'.pkl'
            
            policyWeights = {}
            policy = joblib.load(sourceFile)['policy']
            policyWeights['mean_params'] = sess.run(policy.mean_params)
            policyWeights['std_params'] = sess.run(policy.std_params)

            pickle.dump(policyWeights, open(targetFile , 'wb'))

              