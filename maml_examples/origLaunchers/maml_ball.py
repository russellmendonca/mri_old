from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from maml_examples.point_env_randgoal import PointEnvRandGoal
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from multiworld.envs.mujoco.sawyer_xyz.pick.sawyer_pick_resetArgs import SawyerPickEnv

from multiworld.envs.mujoco.pointMass.ball_resetArgs import BallEnv
from multiworld.core.flat_goal_env import FlatGoalEnv



import tensorflow as tf



fast_learning_rate = 0.1
baseline = 'linear'
fast_batch_size = 20  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 20  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 100
num_grad_updates = 1
meta_step_size = 0.01

use_maml = True


stub(globals())

env = TfEnv(BallEnv())


policy = MAMLGaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    grad_step_size=fast_learning_rate,
    hidden_nonlinearity=tf.nn.relu,
    hidden_sizes=(100,100),
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = MAMLTRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=fast_batch_size, # number of trajs for grad update
    max_path_length=max_path_length,
    meta_batch_size=meta_batch_size,
    num_grad_updates=num_grad_updates,
    n_itr=1000,
    use_maml=use_maml,
    step_size=meta_step_size,
    plot=False,
)

run_experiment_lite(
    algo.train(),
    n_parallel=1,
    snapshot_mode="all",
    python_command='python3',
    seed=1,
    exp_prefix='ball-trial',
    exp_name='trpomaml'+str(int(use_maml))+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates),
    plot=False,
)
