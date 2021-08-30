from pyrep import PyRep
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor

import random
import math
import numpy as np
import random

from config import cfg, cfg_from_file
from utils import get_category


class Env(object):
    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(cfg.SIM.SCENE_FILE, headless=cfg.HEADLESS)
        self.pr.start()
        self.robot = Object.get_object("pepper_base_link_visual")
        self.initial_robot_positions = self.robot.get_position()
        self.initial_robot_orientation = self.robot.get_orientation()
        self.vision_sensor = VisionSensor("Vision_sensor")
        self.vision_sensor.set_resolution(cfg.SIM.VISION_RESOLUTION)

        # clean table
        for obj_class in cfg.SIM.ALL_OBJECTS:
            for obj in obj_class:
                self.set_invisible(Object.get_object(obj))

        # for table scenes
        self.robot_radius = cfg.SIM.ROBOT_RADIUS  # in meters
        self.table_radius = cfg.SIM.TABLE_RADIUS  # in meters
        self.object_table_margin = cfg.SIM.OBJECT_TABLE_MARGIN  # in meters

        # for robot
        self.step_radians = cfg.TRAIN.STEP_RADIANS

        # for occlusion detection
        self.table_height = cfg.SIM.TABLE_HEIGHT  # in meters
        self.robot_camera_height = cfg.SIM.ROBOT_CAM_HEIGHT

    def get_rgbimage(self):
        rgb_image = (
            self.vision_sensor.capture_rgb()
        )  # a ndarray of shape (width, height, 3)
        return rgb_image

    def reset(self, pre_obj_set, obj_set, obj_number, if_occlusion):
        # reset the scene of the previous episode
        for obj in pre_obj_set:
            if obj != "":
                obj_handle = Object.get_object(obj)
                self.set_invisible(obj_handle)

        self.generate_scene(
            obj_set=obj_set, obj_number=obj_number, if_occlusion=if_occlusion
        )
        self.robot.set_position(self.initial_robot_positions)
        self.robot.set_orientation(self.initial_robot_orientation)
        self.pr.step()

    def set_invisible(self, target_obj):
        obj_position = target_obj.get_position()
        obj_position[2] = -1.5
        target_obj.set_position(obj_position)

    def generate_scene(self, obj_set, obj_number, if_occlusion):
        self.obj_number = obj_number
        self.if_occlusion = if_occlusion

        def __check_collision(obj1, obj2):
            obj1_position = obj1.get_position()
            obj1_bounding = obj1.get_bounding_box()
            obj1_r = (abs(obj1_bounding[0]) + abs(obj1_bounding[2])) / 2
            obj2_position = obj2.get_position()
            obj2_bounding = obj2.get_bounding_box()
            obj2_r = (abs(obj2_bounding[0]) + abs(obj2_bounding[2])) / 2

            D = math.sqrt(
                (obj1_position[0] - obj2_position[0]) ** 2
                + (obj1_position[1] - obj2_position[1]) ** 2
            )
            if_too_close = D < obj1_r or D < obj2_r
            is_collision = Object.check_collision(obj1, obj2)
            return if_too_close or is_collision

        def __generate_two_obj_scene(obj1, obj2, obj1_name, obj2_name, occlus):
            # set position
            radius_object_pos = self.table_radius - self.object_table_margin
            big_pos_x, big_pos_y = __generate_pos(radius_object_pos)
            big_pos_z = cfg.SIM.TABLE_HEIGHT + abs(obj1.get_bounding_box()[4])
            obj1.set_position([big_pos_x, big_pos_y, big_pos_z])

            small_pos_x, small_pos_y = __generate_pos(radius_object_pos)
            small_pos_z = cfg.SIM.TABLE_HEIGHT + abs(obj2.get_bounding_box()[4])
            obj2.set_position([small_pos_x, small_pos_y, small_pos_z])

            # set orientation
            orientation_x = obj1.get_orientation()[0]
            orientation_y = obj1.get_orientation()[1]
            if occlus == True and obj1_name == "laptop":
                sign = np.random.choice([1, -1])
                orientation_z = random.uniform(
                    sign * math.pi / 4, sign * 3 * math.pi / 4
                )  # from -pi to pi
            else:
                orientation_z = random.uniform(-math.pi, math.pi)  # -pi to pi
            obj1.set_orientation(
                [
                    orientation_x,
                    orientation_y,
                    orientation_z,
                ]
            )

            orientation_x = obj2.get_orientation()[0]
            orientation_y = obj2.get_orientation()[1]
            if occlus == True and obj2_name == "laptop":
                sign = np.random.choice([1, -1])
                orientation_z = random.uniform(
                    sign * math.pi / 4, sign * 3 * math.pi / 4
                )  # -pi to pi
            else:
                orientation_z = random.uniform(-math.pi, math.pi)  # -pi to pi
            obj2.set_orientation(
                [
                    orientation_x,
                    orientation_y,
                    orientation_z,
                ]
            )

        def __generate_two_obj_scene_without_collision(
            obj1, obj2, obj1_name, obj2_name, occlus
        ):
            # avoid collision
            __generate_two_obj_scene(obj1, obj2, obj1_name, obj2_name, occlus)
            is_collision = __check_collision(obj1, obj2)
            while is_collision:
                __generate_two_obj_scene(obj1, obj2, obj1_name, obj2_name, occlus)
                is_collision = __check_collision(obj1, obj2)

        def __generate_pos(r):
            POS_MIN, POS_MAX = [-r, -r], [r, r]
            pos_x, pos_y = list(np.random.uniform(POS_MIN, POS_MAX))
            while math.pow(pos_x, 2) + math.pow(pos_y, 2) > math.pow(r, 2):
                pos_x, pos_y = list(np.random.uniform(POS_MIN, POS_MAX))
            return pos_x, pos_y

        def __check_occlusion(big_obj, small_obj, big_obj_name, small_obj_name):
            # Obtain layers designed specially for occlusion checking of objects which can not be modeled well by a cylinder
            if big_obj_name == "laptop":
                big_obj_check_occlusion = Object.get_object("laptop_check_occlusion")
            elif big_obj_name == "stuffed_animal":
                big_obj_check_occlusion = Object.get_object(
                    "stuffed_animal_check_occlusion"
                )
            else:
                big_obj_check_occlusion = big_obj
            big_bounding = big_obj_check_occlusion.get_bounding_box()
            # Model each object with a cylinder for occlusion checking
            # we treat the big object as a cylinder. Its diameter is the average of its length and width. The height remains.
            # By adjusting the diameter value, we can adjust the criterion for occlusion checking.
            # If we make the diameter value smaller, the occlusion criterion is stricter, so it is less likely to have partial occlusion situations.
            big_diameter = (abs(big_bounding[0]) * 2 + abs(big_bounding[2]) * 2) / 2
            big_r = big_diameter / 2

            big_position = big_obj_check_occlusion.get_position()
            small_position = small_obj.get_position()

            D = math.sqrt(
                (self.initial_robot_positions[0] - big_position[0]) ** 2
                + (self.initial_robot_positions[1] - big_position[1]) ** 2
            )  # distance from robot to the big object

            # Geometric calculations for occlusion checking
            alpha = math.asin(big_r / D)
            beta = math.asin(abs(big_position[1]) / D)
            theta = beta - alpha
            gamma = beta + alpha

            xa = big_position[0] - big_r * math.sin(
                theta
            )  # (xa,ya), (xb,yb) are two points of tangency
            xb = big_position[0] + big_r * math.sin(gamma)
            if big_position[1] < 0:
                ya = big_position[1] + big_r * math.cos(theta)
                yb = big_position[1] - big_r * math.cos(gamma)
            else:
                ya = big_position[1] - big_r * math.cos(theta)
                yb = big_position[1] + big_r * math.cos(gamma)

            h1 = self.robot_camera_height - self.table_height
            h2 = 2 * abs(big_bounding[4])  # hight of the big object
            w1_biggest = D / (
                1 - h2 / h1
            )  # the maximum distance from the small object to the robot when the small object can be occluded.

            w1_true = math.sqrt(
                (self.initial_robot_positions[0] - small_position[0]) ** 2
                + (self.initial_robot_positions[1] - small_position[1]) ** 2
            )  # the distance from the randomly placed small object to the robot

            occlusion_distance = (
                w1_true > D and w1_true < w1_biggest
            )  # check if the distance is proper for occlusion

            assert self.initial_robot_positions[0] == 1
            assert self.initial_robot_positions[1] == 0

            if big_position[1] < 0:
                xh = xa
                yh = ya
                xl = xb
                yl = yb
            else:
                xh = xb
                yh = yb
                xl = xa
                yl = ya

            occlusion_angle = small_position[1] > yl / (xl - 1) * (
                small_position[0] - 1
            ) and small_position[1] < yh / (xh - 1) * (
                small_position[0] - 1
            )  # check if the relative angle is proper for occlusion; the possible occlusion area is like a sector

            if occlusion_distance and occlusion_angle:
                return True
            else:
                return False

        def generate_one_obj_scene(vis_obj):
            # set position
            radius_object_pos = self.table_radius - self.object_table_margin
            pos_x, pos_y = __generate_pos(radius_object_pos)
            pos_z = cfg.SIM.TABLE_HEIGHT + abs(vis_obj.get_bounding_box()[4])
            vis_obj.set_position([pos_x, pos_y, pos_z])

            # set orientation
            orientation_x = vis_obj.get_orientation()[0]
            orientation_y = vis_obj.get_orientation()[1]
            orientation_z = random.uniform(-math.pi, math.pi)  # -pi to pi
            vis_obj.set_orientation(
                [
                    orientation_x,
                    orientation_y,
                    orientation_z,
                ]
            )

        def generate_two_obj_scene_with_occlusion(big_obj_name, small_obj_name):
            occlus = True
            big_obj = Object.get_object(big_obj_name)
            small_obj = Object.get_object(small_obj_name)
            __generate_two_obj_scene_without_collision(
                big_obj, small_obj, big_obj_name, small_obj_name, occlus
            )
            is_occlusion = __check_occlusion(
                big_obj, small_obj, big_obj_name, small_obj_name
            )
            while not is_occlusion:
                __generate_two_obj_scene_without_collision(
                    big_obj, small_obj, big_obj_name, small_obj_name, occlus
                )
                is_occlusion = __check_occlusion(
                    big_obj, small_obj, big_obj_name, small_obj_name
                )

        def generate_two_obj_scene_without_occlusion(big_obj_name, small_obj_name):
            occlus = False
            big_obj = Object.get_object(big_obj_name)
            small_obj = Object.get_object(small_obj_name)
            __generate_two_obj_scene_without_collision(
                big_obj, small_obj, big_obj_name, small_obj_name, occlus
            )
            is_occlusion = __check_occlusion(
                big_obj, small_obj, big_obj_name, small_obj_name
            )
            while is_occlusion:
                __generate_two_obj_scene_without_collision(
                    big_obj, small_obj, big_obj_name, small_obj_name, occlus
                )
                is_occlusion = __check_occlusion(
                    big_obj, small_obj, big_obj_name, small_obj_name
                )

        # Not used
        def generate_three_obj_scene_with_occlusion(occlus_pair, obj_left_name):
            occlus = True
            big_obj_name = occlus_pair[0]
            small_obj_name = occlus_pair[1]
            big_obj = Object.get_object(big_obj_name)
            small_obj = Object.get_object(small_obj_name)
            # put the two objects from of the occlusion pair.
            __generate_two_obj_scene_without_collision(
                big_obj, small_obj, big_obj_name, small_obj_name, occlus
            )
            is_occlusion = __check_occlusion(
                big_obj, small_obj, big_obj_name, small_obj_name
            )
            while not is_occlusion:
                __generate_two_obj_scene_without_collision(
                    big_obj, small_obj, big_obj_name, small_obj_name, occlus
                )
                is_occlusion = __check_occlusion(
                    big_obj, small_obj, big_obj_name, small_obj_name
                )

            # put the last object
            obj_left = Object.get_object(obj_left_name)
            generate_one_obj_scene(obj_left)
            is_collision_with_big = __check_collision(obj_left, big_obj)
            is_collision_with_small = __check_collision(obj_left, small_obj)
            while is_collision_with_big or is_collision_with_small:
                generate_one_obj_scene(obj_left)
                is_collision_with_big = __check_collision(obj_left, big_obj)
                is_collision_with_small = __check_collision(obj_left, small_obj)

        # Not used
        def generate_three_obj_scene_without_occlusion(obj1_name, obj2_name, obj3_name):
            occlus = False
            obj1 = Object.get_object(obj1_name)
            obj2 = Object.get_object(obj2_name)
            obj3 = Object.get_object(obj3_name)
            # put the first two objects
            __generate_two_obj_scene_without_collision(
                obj1, obj2, obj1_name, obj2_name, occlus
            )
            is_occlusion = __check_occlusion(obj1, obj2, obj1_name, obj2_name)
            while is_occlusion:
                __generate_two_obj_scene_without_collision(
                    obj1, obj2, obj1_name, obj2_name, occlus
                )
                is_occlusion = __check_occlusion(obj1, obj2, obj1_name, obj2_name)

            # put the last object
            generate_one_obj_scene(obj3)
            is_collision_with_obj1 = __check_collision(obj1, obj3)
            is_collision_with_obj2 = __check_collision(obj2, obj3)
            is_occlusion_with_obj1 = __check_occlusion(obj1, obj3, obj1_name, obj3_name)
            is_occlusion_with_obj2 = __check_occlusion(obj2, obj3, obj2_name, obj3_name)
            while (
                is_collision_with_obj1
                or is_collision_with_obj2
                or is_occlusion_with_obj1
                or is_occlusion_with_obj2
            ):
                generate_one_obj_scene(obj3)
                is_collision_with_obj1 = __check_collision(obj1, obj3)
                is_collision_with_obj2 = __check_collision(obj2, obj3)
                is_occlusion_with_obj1 = __check_occlusion(
                    obj1, obj3, obj1_name, obj3_name
                )
                is_occlusion_with_obj2 = __check_occlusion(
                    obj2, obj3, obj2_name, obj3_name
                )

        if self.obj_number == "one":
            generate_one_obj_scene(vis_obj=Object.get_object(obj_set[0]))
        elif self.obj_number == "two":
            # assign big_obj_name and small_obj_name.
            if obj_set[0] in cfg.TRAIN.LARGE_OBJECTS:
                big_obj_name = obj_set[0]
                small_obj_name = obj_set[1]
            elif obj_set[0] in cfg.TRAIN.MEDIUM_OBJECTS:
                if obj_set[1] in cfg.TRAIN.LARGE_OBJECTS:
                    big_obj_name = obj_set[1]
                    small_obj_name = obj_set[0]
                elif obj_set[1] in cfg.TRAIN.SMALL_OBJECTS:
                    big_obj_name = obj_set[0]
                    small_obj_name = obj_set[1]
                else:
                    big_obj_name = obj_set[0]
                    small_obj_name = obj_set[1]
            else:
                big_obj_name = obj_set[1]
                small_obj_name = obj_set[0]

            if (
                self.if_occlusion == "no"
            ):  # It is possible that big_obj_name and small_obj_name are from the same size category.
                generate_two_obj_scene_without_occlusion(big_obj_name, small_obj_name)
            elif (
                self.if_occlusion == "yes"
            ):  # big_objand small_obj must be from different categories.
                generate_two_obj_scene_with_occlusion(big_obj_name, small_obj_name)
            else:
                raise Exception("Cannot generate the scene")

        elif self.obj_number == "three":
            random.shuffle(obj_set)
            obj_set_category = [get_category(obj) for obj in obj_set]
            obj_set_sorted = [
                x
                for _, x in sorted(
                    zip(obj_set_category, obj_set),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ]  # sort the obj_set in a descending order based on the category number.
            if self.if_occlusion == "no":
                generate_three_obj_scene_without_occlusion(
                    obj_set_sorted[0], obj_set_sorted[1], obj_set_sorted[2]
                )
                pass
            elif self.if_occlusion == "yes":
                # three portantial occlusion pairs
                potential_occlus_pairs = [
                    [obj_set_sorted[0], obj_set_sorted[1]],
                    [obj_set_sorted[0], obj_set_sorted[2]],
                    [obj_set_sorted[1], obj_set_sorted[2]],
                ]
                # randomly select one occlusion pair for real occlusion in the scene.
                occlus_pair_idx = np.random.choice(range(3))
                occlus_pair = potential_occlus_pairs[occlus_pair_idx]
                while get_category(occlus_pair[0]) == get_category(
                    occlus_pair[1]
                ):  # make sure to select one pair in which the two objects are from different categories.
                    occlus_pair_idx = np.random.choice(range(3))
                    occlus_pair = potential_occlus_pairs[occlus_pair_idx]
                obj_left = list(set(obj_set_sorted) - set(occlus_pair))[0]
                # make the scene
                generate_three_obj_scene_with_occlusion(occlus_pair, obj_left)
                pass
            else:
                raise Exception("Cannot generate the scene")
            pass
        else:
            raise Exception("Cannot generate the scene")

    def agent_circling(self, direction):  # direction: "left"/"right"
        assert direction == "left" or direction == "right"
        agent_pos = self.robot.get_position()
        radians = math.acos(agent_pos[0] / self.robot_radius)
        if agent_pos[1] < 0:
            radians = -radians
        if direction == "left":
            radians -= self.step_radians
        else:
            radians += self.step_radians
        x = self.robot_radius * math.cos(radians)
        y = self.robot_radius * math.sin(radians)
        self.robot.set_position([x, y, agent_pos[2]])
        self.robot.set_orientation([0, 0, radians - math.pi])
        self.pr.step()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
