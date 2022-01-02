import pickle
import numpy as np

from multiprocessing import Pool

with open("expert.pkl", "rb") as f:
    experts_trajectory = pickle.load(f)


def chunks(arr, n):
    return [arr[i:i + n] for i in range(0, len(arr), n)]


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
out1 = np.where(experts > 100)
exp = np.delete(experts, out1[0], 0)
out2 = np.where(exp < -100)
exp = np.delete(exp, out2[0], 0)
np.random.shuffle(exp)
np.save('experts.npy', exp)
print('husky')
