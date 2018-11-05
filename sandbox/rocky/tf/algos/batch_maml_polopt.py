# import matplotlib
# matplotlib.use('Pdf')

# import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.spaces import Discrete
from rllab.sampler.stateful_pool import singleton_pool
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
import pickle
import joblib

class BatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            # Note that the number of trajectories for grad upate = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            batch_size=100,
            max_path_length=500,
            meta_batch_size = 100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            use_maml=True,
            load_policy=None,
            numExpertPolicies = 10,
            expert_weight= 1,
            expertDataInfo = {},
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  #
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.load_policy=load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size is the number of trajectories for one fast grad update.
        # self.batch_size is the number of total transitions to collect.
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size # number of tasks
        self.num_grad_updates = num_grad_updates # number of gradient steps during training
        self.expertPolicies = {}
        self.numExpertPolicies = numExpertPolicies
        self.a1 = expert_weight
        self.expertDataLoc = expertDataInfo['expert_loc']
        self.expertDataItr = expertDataInfo['expert_itr']

        
        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BatchSampler
            else:
                sampler_cls = VectorizedSampler
        #assert singleton_pool.n_parallel == 1
       

        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix=''):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def process_samples(self, itr, paths, prefix='', log=True, dictPaths = None):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log, dictPaths = dictPaths)


    def expert_logL_bootstrap(self, samples_data , learner_env_goals):

        #returnDict = {}
        for i, expertNum in enumerate(learner_env_goals):
            policy = self.expertPolicies[expertNum]
            expert_logL_traces = policy.compute_stateAction_log_likelihood(samples_data[i]["observations"], samples_data[i]["actions"])
            samples_data[i]['rewards'] =  expert_logL_traces 
            samples_data[i]['rl_rewards'] = samples_data[i]['rewards']
        return samples_data


    def load_expert_trajectories(self):

        import joblib
        expertTrajDir = '/home/russellm/iclr18/data/expert_trajs/Expert_trajs_pusherV1_numTrajs100_itr300/'
        self.expertTrajs = {}
        for task in range(20):
            self.expertTrajs[task] = joblib.load(expertTrajDir+str(task)+'.pkl')
      
    def load_expert_policies(self, sess):

       
        for task in range(self.numExpertPolicies):
          
            print("######LOADING EXPERT "+str(task)+"##############")
            
            policy = GaussianMLPPolicy(name='expert'+str(task), env_spec=self.env.spec, hidden_nonlinearity=tf.nn.relu, hidden_sizes=(100, 100))
            weights  = pickle.load(open(self.expertDataLoc+"Task_"+str(task)+"/itr_"+str(self.expertDataItr)+".pkl" , 'rb'))
            for key in policy.mean_params:
                sess.run(tf.assign(policy.mean_params[key], weights['mean_params'][key]))
            sess.run(tf.assign(policy.std_params , weights['std_params']))
            self.expertPolicies[task] = policy

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
    
        with tf.Session(config = config) as sess:
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            # if self.load_policy is not None:               
            #     self.policy = joblib.load(self.load_policy)['policy']

            self.load_expert_policies(sess)
            self.load_expert_trajectories()
            self.init_opt()
            self.init_opt_offPolicy()
         
            uninit_vars = []
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))

            self.start_worker()
            start_time = time.time()

          
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Sampling set of tasks/goals for this meta-batch...")

                    env = self.env
                    while 'sample_goals' not in dir(env):
                        env = env.wrapped_env
                    learner_env_goals = env.sample_goals(self.meta_batch_size)
                    self.policy.switch_to_init_dist()  # Switch to pre-update policy

                    all_samples_data, all_paths = [], []
                    assert self.num_grad_updates == 1

                    #OnPolicy Iteration
                    if (itr+1)%3 == 0:

                        for step in range(2):
                      
                            logger.log('** Step ' + str(step) + ' **')
                            logger.log("Obtaining samples...")
                            paths = self.obtain_samples(itr, reset_args=learner_env_goals, log_prefix=str(step))
                            all_paths.append(paths)
                            logger.log("Processing samples...")
                            samples_data = {}
                            for key in paths.keys():  # the keys are the tasks
                                # don't log because this will spam the consol with every task.
                                samples_data[key] = self.process_samples(itr, paths[key], log=False)
                            all_samples_data.append(samples_data)

                            # for logging purposes only  
                            self.process_samples(itr, flatten_list(paths.values()), dictPaths = paths, prefix=str(step), log=True)
                            logger.log("Logging diagnostics...")
                            self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))

                            if step == 0:
                                logger.log("Computing policy updates...")
                                self.policy.compute_updated_dists(samples_data)

                            elif step ==1:
                                samples_data = self.expert_logL_bootstrap(samples_data , learner_env_goals)

                        logger.log("Optimizing policy...")
                        dagger = False
                        if dagger:
                            self.optimize_policy(itr, all_samples_data)

                    else:
                        logger.log('Sampling once from meta-Policy')
                        logger.log("Obtaining samples...")
                        paths = self.obtain_samples(itr, reset_args=learner_env_goals, log_prefix='0')
                        all_paths.append(paths)
                        logger.log("Processing samples...")
                        samples_data = {}

                        for key in paths.keys():  # the keys are the tasks
                            # don't log because this will spam the consol with every task.
                            samples_data[key] = self.process_samples(itr, paths[key], log=False)
                        all_samples_data.append(samples_data)
    
                        initialTheta_logProbs = np.array([ self.policy.compute_preUpdatePolicy_logLikelihood(samples_data[key]['observations'] , samples_data[key]['actions']) for key in samples_data])
                    
                        num_imSteps = 20
                        logger.log('Off-policy Optimization')
                        for i in range(num_imSteps):
                            print('Step '+str(i))

                            currTheta_logProbs = np.array([ self.policy.compute_preUpdatePolicy_logLikelihood(samples_data[key]['observations'] , samples_data[key]['actions']) for key in samples_data])
                            traj_imp_weights = self.compute_traj_imp_weights(np.exp(currTheta_logProbs - initialTheta_logProbs))
                            expert_data = {}
                            for i , key in enumerate(learner_env_goals):
                                expert_data[i] = self.process_samples(itr, self.expertTrajs[key], log = False)
                                samples_data[i]['traj_imp_weights'] = traj_imp_weights[i]

                          
                            self.offPolicy_optimization_step(samples_data , expert_data)

                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data[-1]["paths"]
                    logger.save_itr_params(itr, params)

                   
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    logger.dump_tabular(with_prefix=False)

                       
                     
        self.shutdown_worker()

    def compute_traj_imp_weights(self, imp_weights):
        result = [] ; product = 1
        for i in range(len(imp_weights)):
            product*=imp_weights[i]
            result.append(product)
        return np.array(result)
            

    def log_diagnostics(self, paths, prefix):

        

        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
