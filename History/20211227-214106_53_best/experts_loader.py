import pickle
import numpy as np
from utils_psgail import obs_extractor_new

from multiprocessing import Pool

with open("expert.pkl", "rb") as f:
    experts_trajectory = pickle.load(f)


def chunks(arr, n):
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def load_obs(observations):
    vectors = []
    for observation in observations:
        for obs in observation:
            vectors.append(obs_extractor_new(obs))
    return vectors


actions = experts_trajectory['actions']
observations = experts_trajectory['observations']
# pool = Pool(processes=12)
# result = []
# for i in range(12):
#     result.append(pool.apply_async(load_obs, args=(observations[i],)))
# pool.close()
# pool.join()
# vectors = []
# for i in result:
#     vectors += i.get()

for idx, i in enumerate(observations):
    observations[idx] = np.array(i)

np.random.seed(10)
experts_obs = np.vstack(observations)
experts_actions = np.vstack(actions)
experts = np.hstack((experts_obs, experts_actions))
np.random.shuffle(experts)
np.save('experts.npy', experts)
print('husky')
