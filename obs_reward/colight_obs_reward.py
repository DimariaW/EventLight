# Third party code
#
# The following code are copied or modified from:
# https://github.com/zhc134/tlc-baselines and https://github.com/gjzheng93/tlc-baseline

import numpy as np


class ColightGenerator(object):
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay" and "pressure"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "road" means take average of lanes on each road
        "all" means take average of all lanes
    negative : boolean, whether return negative values (mostly for Reward)
    """
    def __init__(self, world, fns_obs, fns_reward, in_only=True, average='road'):

        self.world = world
        self.fns_obs = fns_obs
        # get all the intersections
        self.Is = self.world.intersections
        # get lanes of intersections, with the order of the list of self.Is
        self.all_intersections_lanes = []
        self.obs_dims = []
        for I in self.Is:
            # each intersection's lane_ids is saved in the lanes, and the infos needed for obs can be got from the lane_ids here.
            lanes = []
            # road_ids
            if in_only:
                roads = I.in_roads
            else:
                roads = I.roads
            # get the lane_ids from the road_ids
            for road in roads:
                from_zero = (road["startIntersection"] == I.id
                             ) if self.world.RIGHT else (
                                 road["endIntersection"] == I.id)
                lanes.append([
                    road["id"] + "_" + str(i)
                    for i in range(len(road["lanes"]))[::(
                        1 if from_zero else -1)]
                ])
            # all the lanes of the all the intersections are saved in the self.all_intersections_lanes
            self.all_intersections_lanes.append(lanes)
            # calculate result dim of obs of each agents
            size = sum(len(x) for x in lanes)
            if average == "road":
                size = len(roads)
            elif average == "all":
                size = 1

            self.obs_dims.append(len(self.fns_obs) * size + 8)
            #print("OOOOOOOOOOOOOOO", self.obs_dims[-1])
        # subscribe functions for obs and reward
        self.world.subscribe(self.fns_obs)

        self.world.subscribe(fns_reward)
        self.fns_reward = fns_reward
        self.average = average

    def generate_obs(self):
        """
        return: numpy array of all the intersections obs
        assert that each lane's dim is same.
        """
        # get all the infos for calc the obs of each intersections
        results = [self.world.get_info(fn) for fn in self.fns_obs]

        cur_phases = [I.current_phase for I in self.Is]
        ret_all = []
        for I_id, lanes in enumerate(self.all_intersections_lanes):
            ret = np.array([])
            for i in range(len(self.fns_obs)):
                result = results[i]
                fn_result = np.array([])
                for road_lanes in lanes:
                    road_result = []
                    for lane_id in road_lanes:
                        road_result.append(result[lane_id])
                    if self.average == "road" or self.average == "all":
                        road_result = np.mean(road_result)
                    else:
                        road_result = np.array(road_result)
                    fn_result = np.append(fn_result, road_result)
                if self.average == "all":
                    fn_result = np.mean(fn_result)
                ret = np.append(ret, fn_result)
                # append cur_phase in the last.
                onehot_phase = np.array([1 if i == cur_phases[I_id] else 0 for i in range(8)])
                ret = np.append(ret, onehot_phase)
            ret_all.append(ret)

        return np.array(ret_all)

    def generate_reward(self):
        """
        getting the reward of each intersections, using the pressure.
        """
        pressures = self.world.get_info(self.fns_reward[0])
        rewards = []
        for I in self.world.intersections:
            rewards.append(-pressures[I.id])
        return rewards
    #def generate_reward(self):
    #    """
    #    getting the reward of each intersections, using the pressure.
    #    """
    #    vehicle_reward = []
    #    vehicle_nums = self.world.get_info(self.fns_reward[0])

    #    for I in self.world.intersections:
    #        nvehicles=0
    #        tmp_vehicle = []
    #        in_lanes=[]

    #        for road in I.in_roads:
    #            from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == i.id)
    #            for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
    #                in_lanes.append(road["id"] + "_" + str(n))
    #        

    #        for lane in vehicle_nums.keys():
    #            if lane in in_lanes:
    #                nvehicles += vehicle_nums[lane]
    #                tmp_vehicle.append(vehicle_nums[lane])

    #        tmp_vehicle = np.array(tmp_vehicle)
    #        vehicle_reward.append(-tmp_vehicle.sum()) #return the average length of a intersection
    #    vehicle_reward = np.array(vehicle_reward)
    #    return vehicle_reward

if __name__ == "__main__":

    from world import World
    world = World("./examples/config.json", thread_num=1)
    PressureLightGenerator = ColightGenerator(world, ["lane_count"],
                                                    ["pressure"], False, None)
    for _ in range(200):
        world.step()
        if _ % 50 == 0:
            print(PressureLightGenerator.generate_obs())
            print(PressureLightGenerator.generate_reward())
