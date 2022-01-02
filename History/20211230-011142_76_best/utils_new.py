"""
This file implements the calculation of available features independently. For usage, you should call
`subscribe_features` firstly, then retrive the corresponding observation adapter by define observation space

observation_space = gym.spaces.Dict(subscribe_features(`
    dict(
        distance_to_center=(stack_size, 1),
        speed=(stack_size, 1),
        steering=(stack_size, 1),
        heading_errors=(stack_size, look_ahead),
        ego_lane_dist_and_speed=(stack_size, observe_lane_num + 1),
        img_gray=(stack_size, img_resolution, img_resolution),
    )
))

obs_adapter = get_observation_adapter(
    observation_space,
    look_ahead=look_ahead,
    observe_lane_num=observe_lane_num,
    resize=(img_resolution, img_resolution),
)

"""
import math
import gym
import cv2
import numpy as np

from collections import namedtuple

from smarts.core.sensors import Observation
from smarts.core.utils.math import vec_2d, radians_to_vec
from smarts.core.plan import Start
from smarts.core.coordinates import Heading

Config = namedtuple(
    "Config", "name, agent, interface, policy, learning, other, trainer"
)
FeatureMetaInfo = namedtuple("FeatureMetaInfo", "space, data")

SPACE_LIB = dict(
    distance_to_center=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
    heading_errors=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
    speed=lambda shape: gym.spaces.Box(low=-330.0, high=330.0, shape=shape),
    steering=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
    neighbor=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
    ego_pos=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
    heading=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
    # ego_lane_dist_and_speed=lambda shape: gym.spaces.Box(
    #     low=-1e2, high=1e2, shape=shape
    # ),
    img_gray=lambda shape: gym.spaces.Box(low=0.0, high=1.0, shape=shape),
)


def _cal_angle(vec):
    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    """将周角分成n个区域，获取每个区域最近的车辆"""
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    half_part = math.pi / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist > 100:
            continue
        # calculate its partitions
        angle = np.arctan2(rel_pos_vec[1], rel_pos_vec[0])
        if angle < 0:
            angle = 2 * math.pi + angle
        if 2 * math.pi - half_part > angle >= 0:
            angle += half_part
        else:
            angle = half_part - (2 * math.pi - angle)
        i = int(angle / partition_size)
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    return groups


def get_headingss(x, y):
    rads = np.arctan2(y, x)
    return rads


def get_real_distance(ego_pos, v_pos, ego_bounding, target_bounding):
    rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
    angle_abs = np.arctan2(abs(rel_pos_vec[1]), abs(rel_pos_vec[0]))
    angle_ego = np.arctan2(ego_bounding.width, ego_bounding.length)
    angle_v = np.arctan2(target_bounding.width, target_bounding.length)
    if angle_ego >= angle_abs:
        dist_ego = ego_bounding.length / np.cos(angle_ego) / 2
    else:
        dist_ego = ego_bounding.width / np.sin(angle_abs) / 2
    if angle_v >= angle_abs:
        dist_v = ego_bounding.length / np.cos(angle_v) / 2
    else:
        dist_v = ego_bounding.width / np.sin(angle_abs) / 2
    real_dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec)) - dist_v - dist_ego
    dist_0 = abs(rel_pos_vec[0]) - (ego_bounding.length + target_bounding.length) / 2
    dist_1 = abs(rel_pos_vec[1]) - (ego_bounding.width + target_bounding.width) / 2
    if dist_1 <= 0:
        dist_closest = dist_0
    elif dist_0 <= 0:
        dist_closest = dist_1
    else:
        dist_closest = np.sqrt(dist_0 ** 2 + dist_1 ** 2)
    return real_dist, dist_closest, np.arctan2(rel_pos_vec[1], rel_pos_vec[0])


def get_side_lane_info(position, heading):
    curve = [-1.5597124433619645, -1.5563403521948302, -1.5484982609760536, -1.5297118970629224,
             -1.5365798573689613, -1.5674406346927465]
    relative_heading = 0
    relaltive_pos = 0
    if position[0] < 20.75:
        relative_heading = heading - curve[0]
        relaltive_pos = position[0] * 0.23 / 20.75 - position[1] - 1.98
    elif 20.75 <= position[0] < 49.11:
        relative_heading = heading - curve[1]
        relaltive_pos = (position[0] - 20.75) * 0.41 / 28.36 - position[1] - 1.75
    elif 49.11 <= position[0] < 85.43:
        relative_heading = heading - curve[2]
        relaltive_pos = (position[0] - 49.11) * 0.81 / 36.32 - position[1] - 1.34
    elif 85.43 <= position[0] < 132.38:
        relative_heading = heading - curve[3]
        relaltive_pos = (position[0] - 85.43) * 1.93 / 46.95 - position[1] - 0.53
    elif 132.38 <= position[0] < 165.10:
        relative_heading = heading - curve[4]
        relaltive_pos = (position[0] - 132.38) * 1.12 / 32.72 - position[1] + 1.40
    elif 165.10 <= position[0]:
        relative_heading = heading - curve[5]
        relaltive_pos = (position[0] - 165.10) * 0.05 / 14.9 - position[1] + 2.52
    return relative_heading, relaltive_pos


class CalObs:
    @staticmethod
    def cal_ego_pos(env_obs: Observation, **kwargs):
        return env_obs.ego_vehicle_state.position[:2]

    @staticmethod
    def cal_heading(env_obs: Observation, **kwargs):
        return np.asarray(float(env_obs.ego_vehicle_state.heading))

    @staticmethod
    def cal_distance_to_center(env_obs: Observation, **kwargs):
        """Calculate the signed distance to the center of the current lane.
        Return a FeatureMetaInfo(space, data) instance
        """

        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_to_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        # TODO(ming): for the case of overwhilm, it will throw error
        norm_dist_from_center = signed_dist_to_center / lane_hwidth

        dist = np.asarray([norm_dist_from_center])
        return dist

    @staticmethod
    def cal_heading_errors(env_obs: Observation, **kwargs):
        look_ahead = kwargs["look_ahead"]
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        closest_path = waypoint_paths[closest_wp.lane_index][:look_ahead]

        heading_errors = [
            math.sin(math.radians(wp.relative_heading(ego.heading)))
            for wp in closest_path
        ]

        if len(heading_errors) < look_ahead:
            last_error = heading_errors[-1]
            heading_errors = heading_errors + [last_error] * (
                    look_ahead - len(heading_errors)
            )

        # assert len(heading_errors) == look_ahead
        return np.asarray(heading_errors)

    @staticmethod
    def cal_speed(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        res = np.asarray([ego.speed])
        return res

    @staticmethod
    def cal_steering(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        return np.asarray([ego.steering / 45.0])

    @staticmethod
    def cal_neighbor(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 12)
        # dist, speed, ttc, pos
        features = np.zeros((closest_neighbor_num, 5))
        # fill neighbor vehicles into closest_neighboor_num areas
        surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )
        husky = {'gnE05b_0': [-1.5713518822932961, -0.1, 180, 2.87],
                 'gneE01': [-1.571439578989722, -0.2, 310.92, 8.43, 3.69],
                 'gneE51_0': [-1.5715601518831366, -0.1, 130.92, 3.50]}
        husky_idx = {'gneE01': 1, 'gnE05b_0': 2, 'gneE51_0': 3, 'gneE05a_0': 4}
        ego_pos = ego.position[:2]
        ego_heading = np.asarray(float(ego.heading))
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                # existence rel x, rel y, rel heading, speed
                features[i, :] = np.array([-1, 0, 0, 0, 0])
                continue
            else:
                v = v[0]
            pos = v.position[:2]
            heading = np.asarray(float(v.heading))
            speed = np.asarray(v.speed)
            rel0 = pos[0] - ego_pos[0]
            rel1 = pos[1] - ego_pos[1]
            if 0 <= i <= 1 or i == 11 or 5 <= i <= 7:
                if rel0 >= 0:
                    rel0 = rel0 - v[2].length / 2 - ego[2].length / 2
                else:
                    rel0 = rel0 + v[2].length / 2 + ego[2].length / 2
            else:
                if rel1 >= 0:
                    rel1 = rel1 - v[2].width / 2 - ego[2].width / 2
                else:
                    rel1 = rel1 + v[2].width / 2 + ego[2].width / 2
            features[i, :] = np.asarray([1, rel0, rel1, heading - ego_heading, speed - ego.speed])
        features[:, 3] = features[:, 3] * 180 / math.pi
        features = features.reshape((-1,))
        ego_pos = np.zeros(16)
        ego_pos[0] = ego.heading
        ego_pos[1] = ego.speed
        ego_pos[2:4] = ego.angular_velocity[:2]
        ego_pos[4:6] = ego.angular_acceleration[:2]
        ego_pos[6:8] = ego.linear_velocity[:2]
        ego_pos[8:10] = ego.linear_acceleration[:2]
        # 0.00, -1.98       20.75, 0.23 -1.5597124433619645
        # 20.75, -1.75      28.36 0.41  -1.5563403521948302
        # 49.11, -1.34      36.32 0.81  -1.5484982609760536
        # 85.43, -0.53      46.95 1.93  -1.5297118970629224
        # 132.38, 1.40      32.72 1.12  -1.5365798573689613
        # 165.10, 2.52      14.9    0.05    -1.5674406346927465
        # 180.00, 2.57
        if husky_idx.get(ego.lane_id) is not None:
            if husky_idx.get(ego.lane_id) == 4:
                ego_pos[12] = ego.heading - husky['gnE05b_0'][0]
                ego_pos[13] = ego.position[0] * husky['gnE05b_0'][1] / husky['gnE05b_0'][2] + husky['gnE05b_0'][
                    3] - ego.position[1]
                ego_pos[10], ego_pos[11] = get_side_lane_info(ego.position, ego.heading)
                ego_pos[14] = 0
                ego_pos[15] = 0
            elif husky_idx.get(ego.lane_id) == 2:
                ego_pos[12] = ego.heading - husky['gneE01'][0]
                ego_pos[13] = ego.position[0] * husky['gneE01'][1] / husky['gneE01'][2] + husky['gneE01'][
                    3] - ego.position[1]
                ego_pos[10] = ego.heading - husky['gnE05b_0'][0]
                ego_pos[11] = ego.position[0] * husky['gnE05b_0'][1] / husky['gnE05b_0'][2] + husky['gnE05b_0'][
                    3] - ego.position[1]
                ego_pos[14], ego_pos[15] = get_side_lane_info(ego.position, ego.heading)
            elif husky_idx.get(ego.lane_id) == 3:
                ego_pos[12] = ego.heading - husky['gneE01'][0]
                ego_pos[13] = ego.position[0] * husky['gneE01'][1] / husky['gneE01'][2] + husky['gneE01'][
                    3] - ego.position[1]
                ego_pos[10] = ego.heading - husky['gneE51_0'][0]
                ego_pos[11] = (ego.position[0] - 180) * husky['gneE51_0'][1] / husky['gneE51_0'][2] + \
                              husky['gneE51_0'][3] - ego.position[1]
                ego_pos[14] = 0
                ego_pos[15] = 0
        elif ego.lane_id is not None and ego.lane_id[:6] in husky_idx:
            if ego.lane_index == 4:
                ego_pos[12] = 0
                ego_pos[13] = 0
            else:
                ego_pos[12] = ego.heading - husky['gneE01'][0]
                ego_pos[13] = ego.position[0] * husky['gneE01'][1] / husky['gneE01'][2] + \
                              husky['gneE01'][3] + (ego.lane_index + 1) * husky['gneE01'][4] - ego.position[1]
            ego_pos[10] = ego.heading - husky['gneE01'][0]
            ego_pos[11] = ego.position[0] * husky['gneE01'][1] / husky['gneE01'][2] + \
                          husky['gneE01'][3] + ego.lane_index * husky['gneE01'][4] - ego.position[1]
            if ego.lane_index == 0:
                if ego.position[0] <= 180:
                    ego_pos[14] = ego.heading - husky['gnE05b_0'][0]
                    ego_pos[15] = ego.position[0] * husky['gnE05b_0'][1] / husky['gnE05b_0'][2] + husky['gnE05b_0'][
                        3] - ego.position[1]
                else:
                    ego_pos[14] = ego.heading - husky['gneE51_0'][0]
                    ego_pos[15] = (ego.position[0] - 180) * husky['gneE51_0'][1] / husky['gneE51_0'][2] + \
                                  husky['gneE51_0'][3] - ego.position[1]
            else:
                ego_pos[14] = ego.heading - husky['gneE01'][0]
                ego_pos[15] = ego.position[0] * husky['gneE01'][1] / husky['gneE01'][2] + \
                              husky['gneE01'][3] + (ego.lane_index - 1) * husky['gneE01'][4] - ego.position[1]
        ego_pos[[10, 12, 14]] = ego_pos[[10, 12, 14]] * 180 / math.pi
        vecs = np.concatenate((features, ego_pos), axis=0)
        return vecs

    @staticmethod
    def cal_ego_lane_dist_and_speed(env_obs: Observation, **kwargs):
        """Calculate the distance from ego vehicle to its front vehicles (if have) at observed lanes,
        also the relative speed of the front vehicle which positioned at the same lane.
        """
        observe_lane_num = kwargs["observe_lane_num"]
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        wps_with_lane_dist = []
        for path_idx, path in enumerate(waypoint_paths):
            lane_dist = 0.0
            for w1, w2 in zip(path, path[1:]):
                wps_with_lane_dist.append((w1, path_idx, lane_dist))
                lane_dist += np.linalg.norm(w2.pos - w1.pos)
            wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

        # TTC calculation along each path
        ego_closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        wps_on_lane = [
            (wp, path_idx, dist)
            for wp, path_idx, dist in wps_with_lane_dist
            # if wp.lane_id == v.lane_id
        ]

        ego_lane_index = closest_wp.lane_index
        lane_dist_by_path = [1] * len(waypoint_paths)
        ego_lane_dist = [0] * observe_lane_num
        speed_of_closest = 0.0

        for v in env_obs.neighborhood_vehicle_states:
            nearest_wp, path_idx, lane_dist = min(
                wps_on_lane,
                key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position)),
            )
            if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                # this vehicle is not close enough to the path, this can happen
                # if the vehicle is behind the ego, or ahead past the end of
                # the waypoints
                continue

            # relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
            # relative_speed_m_per_s = max(abs(relative_speed_m_per_s), 1e-5)
            dist_wp_vehicle_vector = vec_2d(v.position) - vec_2d(nearest_wp.pos)
            direction_vector = np.array(
                [
                    math.cos(math.radians(nearest_wp.heading)),
                    math.sin(math.radians(nearest_wp.heading)),
                ]
            ).dot(dist_wp_vehicle_vector)

            dist_to_vehicle = lane_dist + np.sign(direction_vector) * (
                np.linalg.norm(vec_2d(nearest_wp.pos) - vec_2d(v.position))
            )
            lane_dist = dist_to_vehicle / 100.0

            if lane_dist_by_path[path_idx] > lane_dist:
                if ego_closest_wp.lane_index == v.lane_index:
                    speed_of_closest = (v.speed - ego.speed) / 120.0

            lane_dist_by_path[path_idx] = min(lane_dist_by_path[path_idx], lane_dist)

        # current lane is centre
        flag = observe_lane_num // 2
        ego_lane_dist[flag] = lane_dist_by_path[ego_lane_index]

        max_lane_index = len(lane_dist_by_path) - 1

        if max_lane_index == 0:
            right_sign, left_sign = 0, 0
        else:
            right_sign = -1 if ego_lane_index + 1 > max_lane_index else 1
            left_sign = -1 if ego_lane_index - 1 >= 0 else 1

        ego_lane_dist[flag + right_sign] = lane_dist_by_path[
            ego_lane_index + right_sign
            ]
        ego_lane_dist[flag + left_sign] = lane_dist_by_path[ego_lane_index + left_sign]

        res = np.asarray(ego_lane_dist + [speed_of_closest])
        return res
        # space = SPACE_LIB["goal_relative_pos"](res.shape)
        # return (res - space.low) / (space.high - space.low)

    @staticmethod
    def cal_img_gray(env_obs: Observation, **kwargs):
        resize = kwargs["resize"]

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        rgb_ndarray = env_obs.top_down_rgb
        gray_scale = (
                cv2.resize(
                    rgb2gray(rgb_ndarray), dsize=resize, interpolation=cv2.INTER_CUBIC
                )
                / 255.0
        )
        return gray_scale


def _update_obs_by_item(
        ith, obs_placeholder: dict, tuned_obs: dict, space_dict: gym.spaces.Dict
):
    for key, value in tuned_obs.items():
        if obs_placeholder.get(key, None) is None:
            obs_placeholder[key] = np.zeros(space_dict[key].shape)
        obs_placeholder[key][ith] = value


def _cal_obs(env_obs: Observation, space, **kwargs):
    obs = dict()
    for name in space.spaces:
        if hasattr(CalObs, f"cal_{name}"):
            obs[name] = getattr(CalObs, f"cal_{name}")(env_obs, **kwargs)
    return obs


def subscribe_features(**kwargs):
    res = dict()

    for k, config in kwargs.items():
        if bool(config):
            res[k] = SPACE_LIB[k](config)

    return res


# XXX(ming): refine it as static method
def get_observation_adapter(observation_space, **kwargs):
    def observation_adapter(env_obs):
        obs = dict()
        if isinstance(env_obs, list) or isinstance(env_obs, tuple):
            for i, e in enumerate(env_obs):
                temp = _cal_obs(e, observation_space, **kwargs)
                _update_obs_by_item(i, obs, temp, observation_space)
        else:
            temp = _cal_obs(env_obs, observation_space, **kwargs)
            _update_obs_by_item(0, obs, temp, observation_space)
        return obs

    return observation_adapter


def get_vehicle_start_at_time(vehicle_id, start_time, traffic_history):
    pphs = traffic_history.vehicle_pose_at_time(vehicle_id, start_time)
    assert pphs
    pos_x, pos_y, heading, speed = pphs
    veh_length, veh_width, veh_height = traffic_history.vehicle_size(str(vehicle_id))
    # missions start from front bumper, but pos is center of vehicle
    hhx, hhy = radians_to_vec(heading) * (0.5 * veh_length)
    return Start(
        (pos_x + hhx, pos_y + hhy),
        Heading(heading),
    )
