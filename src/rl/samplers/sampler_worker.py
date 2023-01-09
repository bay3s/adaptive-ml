import multiprocessing as mp


class SamplerWorker(mp.Process):


  def __init__(self, index, env_name, env_kwargs, batch_size, observation_space, action_space, policy, baseline, seed,
               task_queue, train_queue, valid_queue, policy_lock):

  super(SamplerWorker, self).__init__()

  pass
