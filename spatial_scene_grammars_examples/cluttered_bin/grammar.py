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
Bin -> object (1-N)
object -> `or node of all manipulands, spawned uniformly above the bin`

All objects inside bin constraint
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


class ClutteredBin(RepeatingSetNode):
    bin_dims = torch.tensor([0.44, 0.29, 0.23])
    margin = 0.05
    bin_lower_bounds = torch.tensor(
        [-bin_dims[0] / 2 + margin, -bin_dims[1] / 2 + margin, bin_dims[2]]
    )
    bin_upper_bounds = torch.tensor(
        [bin_dims[0] / 2 - margin, bin_dims[1] / 2 - margin, bin_dims[2] * 3]
    )

    def __init__(self, tf, max_children=10):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(
            torch.eye(4), "package://scalable_real2sim/bin/bin.sdf"
        )

        # Uniform distribution over objects, at least one object.
        rule_probs = torch.ones(max_children) / (max_children - 1)
        rule_probs[0] = 0.0

        super().__init__(
            tf=tf, physics_geometry_info=geom, observed=True, rule_probs=rule_probs
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Object,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    cls.bin_lower_bounds, cls.bin_upper_bounds
                ),
                rotation_rule=SameRotationRule(),
            ),
        ]


class Object(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.ones(20) / 10,
            observed=False,
            physics_geometry_info=None,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=BleachBottle,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=CameraMount,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=CherryJello,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ExpoBrush,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Inflator,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Ketchup,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LargeFinray,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LeafBags,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Lego,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LimeJello,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Mouthwash,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=MustardLarge,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=OrganicMustard,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Pollinator,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Skittles,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SmallMustard,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Spam,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SugarBox,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Tide,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Unitek,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class BleachBottle(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/bleach_bottle/bleach_bottle.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class CameraMount(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/camera_mount/camera_mount.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class CherryJello(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/cherry_jello/cherry_jello.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class ExpoBrush(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/expo_brush/expo_brush.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Inflator(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/inflator/inflator.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Ketchup(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/ketchup/ketchup.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class LargeFinray(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/large_finray/large_finray.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class LeafBags(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/leaf_bags/leaf_bags.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Lego(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/lego/lego.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class LimeJello(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/lime_jello/lime_jello.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Mouthwash(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/mouthwash/mouthwash.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class MustardLarge(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/mustard_large/mustard_large.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class OrganicMustard(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/organic_mustard/organic_mustard.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Pollinator(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/pollinator/pollinator.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Skittles(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/skittles/skittles.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class SmallMustard(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/small_mustard/small_mustard.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Spam(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/spam/spam.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class SugarBox(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/sugar_box/sugar_box.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Tide(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/tide/tide.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Unitek(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://scalable_real2sim/unitek/unitek.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)
