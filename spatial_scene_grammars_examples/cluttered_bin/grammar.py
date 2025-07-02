import glob
import os
from functools import lru_cache

import pydrake
import pydrake.geometry as pydrake_geom
import torch
from pydrake.all import RigidTransform, RollPitchYaw

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars_examples.tri_living_room_shelf.grammar import Shelf

"""
Restaurant -> table (1-10) & shelf (0-3)
Table -> place settings and shared dishware
Shared dishware -> Tea kettle, food plates, bamboo steamer towers
Place settings - > cup, plate, chopsticks, chair?


"""
ARBITRARY_YAW_ROTATION_RULE = (
    ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
        RotationMatrix(), np.array([1e6, 1e6, 1])
    )  # Bigger values = less variance
)
ARBITRARY_ROTATION_RULE = (
    ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
        RotationMatrix(), np.array([1, 1, 1])
    )
)


class ClutteredBin(AndNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(
            torch.eye(4), "package://greg_table/models/misc/floor/model.sdf"
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Tables,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Shelves,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
