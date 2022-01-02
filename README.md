To see the history of development, please visit https://github.com/oldpaws/NGSIM_SMARTS.
![img](https://github.com/oldpaws/AI3601_RL_Final_Project_MAGAIL/blob/main/demo/10_agents_0.gif)


# Benchmarks
```
python multiagent_benchmark_53/benchmark.py ./ngsim --env_num 12
python multiagent_benchmark_76/benchmark.py ./ngsim --env_num 12
```

# Visualize
```
python Visualize/multiagent_visual.py   
```

# Experiments
```
python psgail_train_basic.py
python psgail_train_bc.py
python psgail_train_full.py
```
# experts_53_cleaned.npy
https://jbox.sjtu.edu.cn/l/z1Hq5a (提取码：jung)

# experts_76_new.npy
https://jbox.sjtu.edu.cn/l/z1HQ4y (提取码：vppv)

# Best Agents
Original Code and running logs of the best agents are in `History`

View the full logs with `tensorboard --logdir History/`

# Benchmark Results
psgail_1557_gail_2_260_old_best_53.model

Average Frechet Distance: 63.533723286525706

Average Distance Travelled: 243.27960304940157

Success Rate: 0.595449500554939

---

psgail_1772_gail_2_264_old_best_53.model

Average Frechet Distance: 51.99174296272629

Average Distance Travelled: 255.2405674310502

Success Rate: 0.6514983351831298

---

psgail_1221_gail_2_943_931_941_cpu.pt

Average Frechet Distance: 73.34188714218737

Average Distance Travelled: 232.99353401315335

Success Rate: 0.5457534246575343

---

psgail_1219_gail_2_895_931_967_cpu.pt

Average Frechet Distance: 76.67974239694301

Average Distance Travelled: 229.49846935362635

Success Rate: 0.5461031833150384

---

psgail_1220_gail_2_927_944_920_cpu.pt

Average Frechet Distance: 73.42993106223454

Average Distance Travelled: 232.8953028749918

Success Rate: 0.5578082191780822

---

psgail_986_gail_2_855_869_933.model

Average Frechet Distance: 74.33327177655201

Average Distance Travelled: 231.64199129366705

Success Rate: 0.5605479452054795
