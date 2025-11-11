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
from spatial_scene_grammars_examples.tri_living_room_shelf.grammar import (
    Shelf,
    EmptyShelf,
)

"""
Restaurant -> table (1-10) & shelf (0-3)
Table -> place settings and shared dishware
Shared dishware -> Tea kettle, food plates, bamboo steamer towers
Place settings - > cup, plate, chopsticks, chair?

# TODO: Need to add shelves
# TODO: Add structure constraint for preventing collision between tables and shelves
# TODO: Update greg on Anzu
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


class Teacup(TerminalNode):
    KEEPOUT_RADIUS = 0.07

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg/models/plates_cups_and_bowls/cups/coffee_cup_white/model_simplified.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Teapot(TerminalNode):
    KEEPOUT_RADIUS = 0.1

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg/models/plates_cups_and_bowls/cups/Threshold_Porcelain_Teapot_White/model_simplified.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)


class ClutteredBin(RepeatingSetNode):
    bin_dims = torch.tensor([0.45, 0.3, 0.23])
    margin = 0.1
    bin_lower_bounds = torch.tensor(
        [-bin_dims[0] / 2 + margin, -bin_dims[1] / 2 + margin, bin_dims[2] / 2]
    )
    bin_upper_bounds = torch.tensor(
        [bin_dims[0] / 2 - margin, bin_dims[1] / 2 - margin, bin_dims[2] * 6]
    )

    # For collision detection (similar to Shelf.KEEPOUT_RADIUS)
    KEEPOUT_RADIUS = max(bin_dims[0].item(), bin_dims[1].item()) / 2.0  # ~0.22m

    def __init__(self, tf, min_children=3, max_children=15):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(torch.eye(4), "package://greg/models/misc/bin/bin_coacd.sdf")

        # Uniform distribution over objects, at least min_children objects.
        rule_probs = torch.ones(max_children)
        rule_probs[: min_children - 1] = 0.0
        rule_probs = rule_probs / rule_probs.sum()

        super().__init__(
            tf=tf, physics_geometry_info=geom, observed=True, rule_probs=rule_probs
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Object,
                xyz_rule=ParentFrameBBoxOffsetRule.from_bounds(
                    cls.bin_lower_bounds, cls.bin_upper_bounds
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e2, 1e2, 10])
                ),  # Bigger values = less variance
            ),
        ]


class Object(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.ones(14) / 14,
            observed=False,
            physics_geometry_info=None,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=CerealBox,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Toast,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Apple,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Apple1,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Apple2,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Apple3,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=FuerteAvocado,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=HassAvocado,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LulaAvocado,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=StarkrimsonPear,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=BosePear,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=GreenAnjouPear,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=TeaBottle,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=BananaConcentrate,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class CerealBox(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/food/cereal/punyos_cereal_box.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Toast(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/food/sandwich/fake_toasted_bread_slice_mesh_collision.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Apple(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/apples/gala_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Apple1(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/apples/fake_red_delicious_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Apple2(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/apples/granny_smith_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Apple3(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/apples/golden_delicious_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class FuerteAvocado(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/avocados/fuerte_avocado.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class HassAvocado(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/avocados/hass_avocado.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class LulaAvocado(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/avocados/lula_avocado.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class StarkrimsonPear(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/pears/starkrimson_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class BosePear(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/pears/bose_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class GreenAnjouPear(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/fruits/pears/green_anjou_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class TeaBottle(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/home_kitchen/junk/tea_bottle.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class BananaConcentrate(TerminalNode):

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://tri/models/home_kitchen/junk/rlg_banana_concentrate.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class EmptyClutteredBin(TerminalNode):
    """Empty bin container - just the bin mesh without any internal objects.
    Used for stage 1 sampling of container layout."""

    # Copy class variables from ClutteredBin for collision detection
    bin_dims = torch.tensor([0.45, 0.3, 0.23])
    KEEPOUT_RADIUS = max(bin_dims[0].item(), bin_dims[1].item()) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(torch.eye(4), "package://greg/models/misc/bin/bin.sdf")
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class SteamerBottom(OrNode):
    KEEPOUT_RADIUS = 0.12

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg/models/misc/steamer_bottom/model.sdf",
        )
        super().__init__(
            tf=tf,
            # rule_probs=torch.tensor([0.3, 0.4, 0.3]),
            rule_probs=torch.tensor([0.6, 0.25, 0.15]),
            physics_geometry_info=geom,
            observed=True,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=SteamerBottom,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.091]),
                    variance=torch.tensor([1e-16, 1e-16, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=SteamerTop,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.091]),
                    variance=torch.tensor([1e-16, 1e-16, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class SteamerTop(TerminalNode):
    KEEPOUT_RADIUS = 0.12

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg/models/misc/steamer_top/model.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


TabletopObjectTypes = (Teacup, Teapot, SteamerBottom, SteamerTop)


class SharedTeacups(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.5, max_children=2, start_at_one=True
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Teacup,
                xyz_rule=AnnulusOffsetRule(
                    min_radius=0.35,
                    max_radius=0.7,
                    z_height=0.0,
                    angle_min=-110.0,
                    angle_max=110.0,
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedTeapots(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.5, max_children=2, start_at_one=True
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Teapot,
                xyz_rule=AnnulusOffsetRule(
                    min_radius=0.35,
                    max_radius=0.7,
                    z_height=0.0,
                    angle_min=-110.0,
                    angle_max=110.0,
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedSteamers(RepeatingSetNode):
    def __init__(self, tf):
        # Create a custom distribution that starts at 2 steamers
        rule_probs = torch.zeros(5)  # For 0 to 4 steamers
        rule_probs[0:2] = 0.0  # Zero probability for 0 or 1 steamer

        # Geometric distribution starting at 2
        p = 0.2
        for k in range(2, 5):
            rule_probs[k] = (1 - p) ** (k - 2) * p

        # Normalize to sum to 1
        rule_probs = rule_probs / torch.sum(rule_probs)

        super().__init__(
            tf=tf,
            rule_probs=rule_probs,
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SteamerBottom,
                xyz_rule=AnnulusOffsetRule(
                    min_radius=0.35,
                    max_radius=0.7,
                    z_height=0.0,
                    angle_min=-110.0,
                    angle_max=110.0,
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedStuff(IndependentSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.9, 0.6, 0.9]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SharedTeapots,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedTeacups,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedSteamers,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class FloorObjectsRoot(AndNode):
    """Wrapper node for floor object sampling that applies CircularOffsetRule to SharedStuff.

    Used in hierarchical multi-stage sampling to ensure SharedStuff objects are placed
    in the pie region [-110, 110] at radius 0.9m, matching the original AdamScene behavior.
    """

    SHARED_STUFF_RADIUS = 0.7  # Match AdamScene radius

    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SharedStuff,
                xyz_rule=CircularOffsetRule(
                    radius=cls.SHARED_STUFF_RADIUS,
                    z_height=0.0,
                    angle_min=-110.0,
                    angle_max=110.0,
                ),
                rotation_rule=SameRotationRule(),
            ),
        ]


class Bins(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.3, max_children=3, start_at_one=False
            ),
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=ClutteredBin,
                xyz_rule=CircularOffsetRule(
                    radius=0.6, z_height=0.0, angle_min=-120.0, angle_max=120.0
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class Shelves(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.3, max_children=3, start_at_one=False
            ),
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Shelf,
                xyz_rule=CircularOffsetRule(
                    radius=0.86, z_height=0.404394, angle_min=-110.0, angle_max=110.0
                ),
                rotation_rule=FaceOriginRotationRule(
                    target_point=torch.zeros(3), facing_axis="x"
                ),
            )
        ]


class EmptyShelves(RepeatingSetNode):
    """RepeatingSetNode for empty shelves (stage 1 container layout)."""

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.25, max_children=3, start_at_one=False
            ),
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=EmptyShelf,
                xyz_rule=CircularOffsetRule(
                    radius=0.86, z_height=0.404394, angle_min=-110.0, angle_max=110.0
                ),
                rotation_rule=FaceOriginRotationRule(
                    target_point=torch.zeros(3), facing_axis="x"
                ),
            )
        ]


class EmptyBins(RepeatingSetNode):
    """RepeatingSetNode for empty bins (stage 1 container layout)."""

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.25, max_children=3, start_at_one=False
            ),
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=EmptyClutteredBin,
                xyz_rule=CircularOffsetRule(
                    radius=0.6, z_height=0.0, angle_min=-110.0, angle_max=110.0
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class AdamScene(AndNode):
    SHARED_STUFF_RADIUS = 0.9  # Radius for SharedStuff placement

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(
            torch.eye(4),
            "package://greg/models/misc/iiwa_env.dmd.yaml",
            root_body_name="floor_base",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SharedStuff,
                xyz_rule=CircularOffsetRule(
                    radius=cls.SHARED_STUFF_RADIUS,
                    z_height=0.0,
                    angle_min=-110.0,
                    angle_max=110.0,
                ),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Shelves,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Bins,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class Stage1AdamScene(AndNode):
    """Stage 1 scene for multi-stage sampling: empty bins and shelves only.

    This scene is used in the first stage of hierarchical sampling to determine
    the layout of empty containers (bins and shelves) before populating them
    with objects in stage 2.
    """

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom.register_model_file(
            torch.eye(4),
            "package://greg/models/misc/iiwa_env.dmd.yaml",
            root_body_name="floor_base",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=EmptyShelves,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=EmptyBins,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class TallStackConstraint(StructureConstraint):
    # The largest stack of steamers is at least 3 steamers tall.
    def __init__(self, min_height=3):
        lb = torch.tensor([min_height])
        ub = torch.tensor([np.inf])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        steamers = scene_tree.find_nodes_by_type(SteamerBottom)
        tallest_stack = 0
        # For each steamer, count how many parents it has that
        # are SteamerBottoms before hitting something else.
        # This #+1 is the number of steamers in the stack.
        for steamer in steamers:
            current_steamer = steamer
            stack = 0
            while isinstance(current_steamer, SteamerBottom):
                stack += 1
                current_steamer = scene_tree.get_parent(current_steamer)
            tallest_stack = max(tallest_stack, stack)
        return torch.tensor([tallest_stack])


class ObjectsOutsideIiwa(PoseConstraint):
    def __init__(self, radius=0.4):
        # Constraint on squared distance from origin (iiwa base)
        # Objects must be at least 'radius' meters away in the XY plane
        lb = torch.tensor([radius * radius])
        ub = torch.tensor([np.inf])
        super().__init__(lower_bound=lb, upper_bound=ub)
        self.radius = radius

    def eval(self, scene_tree):
        # Find all observed tabletop objects
        distances_squared = []
        for obj_type in TabletopObjectTypes:
            objs = scene_tree.find_nodes_by_type(obj_type)
            for obj in objs:
                if obj.observed:
                    # Compute squared distance from origin in XY plane only
                    dist_sq = torch.sum(obj.translation[:2] ** 2)
                    distances_squared.append(dist_sq)

        if len(distances_squared) > 0:
            return torch.stack(distances_squared, axis=0).reshape(-1, 1)
        else:
            return torch.empty(size=(0, 1))

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class ObjectsWithinArcConstraint(PoseConstraint):
    """Ensures tabletop objects only appear within a specified angular arc around origin."""

    def __init__(self, angle_min=-110.0, angle_max=110.0):
        # Convert degrees to radians
        self.angle_min_rad = angle_min * np.pi / 180.0
        self.angle_max_rad = angle_max * np.pi / 180.0

        # Lower bound: 0 means object must be within arc
        # Upper bound: infinity (no upper limit on how far inside arc)
        lb = torch.tensor([0.0])
        ub = torch.tensor([np.inf])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        # Find all observed tabletop objects
        margins = []
        for obj_type in TabletopObjectTypes:
            objs = scene_tree.find_nodes_by_type(obj_type)
            for obj in objs:
                if obj.observed:
                    # Compute angle from origin in XY plane
                    angle = torch.atan2(obj.translation[1], obj.translation[0])

                    # Compute margin: how far inside the allowed arc
                    # Positive if within arc, negative if outside
                    # Margin = min(angle - angle_min, angle_max - angle)
                    margin_from_min = angle - self.angle_min_rad
                    margin_from_max = self.angle_max_rad - angle
                    margin = torch.min(margin_from_min, margin_from_max)

                    margins.append(margin)

        if len(margins) > 0:
            return torch.stack(margins, axis=0).reshape(-1, 1)
        else:
            return torch.empty(size=(0, 1))

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class MinNumShelvesAndBinsConstraint(StructureConstraint):
    """Ensures at least a minimum number of shelves and bins are present in the scene."""

    def __init__(self, min_count=2):
        lb = torch.tensor([min_count])
        ub = torch.tensor([np.inf])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        shelves = scene_tree.find_nodes_by_type(Shelf)
        bins = scene_tree.find_nodes_by_type(ClutteredBin)
        # Also check for empty variants (used in stage 1 sampling)
        empty_shelves = scene_tree.find_nodes_by_type(EmptyShelf)
        empty_bins = scene_tree.find_nodes_by_type(EmptyClutteredBin)
        total_count = len(shelves) + len(bins) + len(empty_shelves) + len(empty_bins)

        return torch.tensor([float(total_count)])


class ShelvesNotInCollisionWithBinsConstraint(StructureConstraint):
    """Ensures that shelves don't collide with each other using oriented bounding
    box detection."""

    def __init__(self):
        # Constraint satisfied when all separations are >= 0
        super().__init__(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([np.inf])
        )

    def eval(self, scene_tree):
        # Get both populated and empty variants
        shelves = scene_tree.find_nodes_by_type(Shelf)
        bins = scene_tree.find_nodes_by_type(ClutteredBin)
        empty_shelves = scene_tree.find_nodes_by_type(EmptyShelf)
        empty_bins = scene_tree.find_nodes_by_type(EmptyClutteredBin)

        # Combine into single lists for collision checking
        all_shelves = shelves + empty_shelves
        all_bins = bins + empty_bins

        separations = []

        # Check shelf-shelf collisions using OBB
        for i in range(len(all_shelves)):
            for j in range(i + 1, len(all_shelves)):
                separation = self._check_obb_separation(
                    all_shelves[i],
                    all_shelves[j],
                    Shelf.WIDTH / 2.0,
                    Shelf.LENGTH / 2.0,
                    Shelf.WIDTH / 2.0,
                    Shelf.LENGTH / 2.0,
                )
                separations.append(separation)

        # Check bin-bin collisions using OBB
        for i in range(len(all_bins)):
            for j in range(i + 1, len(all_bins)):
                separation = self._check_obb_separation(
                    all_bins[i],
                    all_bins[j],
                    ClutteredBin.bin_dims[0] / 2.0,
                    ClutteredBin.bin_dims[1] / 2.0,
                    ClutteredBin.bin_dims[0] / 2.0,
                    ClutteredBin.bin_dims[1] / 2.0,
                )
                separations.append(separation)

        # Check shelf-bin collisions using OBB
        for shelf in all_shelves:
            for bin in all_bins:
                separation = self._check_obb_separation(
                    shelf,
                    bin,
                    Shelf.WIDTH / 2.0,
                    Shelf.LENGTH / 2.0,
                    ClutteredBin.bin_dims[0] / 2.0,
                    ClutteredBin.bin_dims[1] / 2.0,
                )
                separations.append(separation)

        if len(separations) == 0:
            return torch.tensor([[1.0]])  # Trivially satisfied, no pairs to check

        return torch.stack(separations).reshape(-1, 1)

    def _check_obb_separation(
        self,
        obj1,
        obj2,
        obj1_half_width,
        obj1_half_length,
        obj2_half_width,
        obj2_half_length,
    ):
        """Check oriented bounding box separation using Separating Axis Theorem (SAT).

        Args:
            obj1, obj2: Objects with .translation and .rotation attributes
            obj1_half_width, obj1_half_length: Half-extents of obj1's bounding box
            obj2_half_width, obj2_half_length: Half-extents of obj2's bounding box

        Returns:
            Minimum separation distance (positive = separated, negative = overlapping)
        """
        # Extract 2D positions and rotations
        pos1 = obj1.translation[:2]
        pos2 = obj2.translation[:2]
        rot1 = obj1.rotation[:2, :2]  # 2x2 XY rotation matrix
        rot2 = obj2.rotation[:2, :2]

        # Compute corners of each object in local frame
        local_corners1 = torch.tensor(
            [
                [obj1_half_width, obj1_half_length],
                [-obj1_half_width, obj1_half_length],
                [-obj1_half_width, -obj1_half_length],
                [obj1_half_width, -obj1_half_length],
            ]
        )

        local_corners2 = torch.tensor(
            [
                [obj2_half_width, obj2_half_length],
                [-obj2_half_width, obj2_half_length],
                [-obj2_half_width, -obj2_half_length],
                [obj2_half_width, -obj2_half_length],
            ]
        )

        # Transform corners to world frame
        corners1 = torch.matmul(rot1, local_corners1.T).T + pos1
        corners2 = torch.matmul(rot2, local_corners2.T).T + pos2

        # SAT: test separation along 4 axes (2 per oriented box)
        # For 2D OBB, we only need to test the face normals of each box
        axes = [
            rot1[:, 0],  # obj1 local X-axis
            rot1[:, 1],  # obj1 local Y-axis
            rot2[:, 0],  # obj2 local X-axis
            rot2[:, 1],  # obj2 local Y-axis
        ]

        min_separation = torch.tensor(float("inf"))

        for axis in axes:
            # Normalize axis
            axis = axis / (torch.norm(axis) + 1e-8)

            # Project all corners onto this axis
            proj1 = torch.matmul(corners1, axis)
            proj2 = torch.matmul(corners2, axis)

            # Get min and max projections for each box
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()

            # Calculate separation on this axis
            # Positive if separated, negative if overlapping
            separation = torch.max(min2 - max1, min1 - max2)

            # Track minimum separation across all axes
            min_separation = torch.min(min_separation, separation)

        return min_separation


class SharedStuffNotInCollisionWithShelvesAndBins(PoseConstraint):
    """Ensures that SharedStuff objects (tabletop items) don't collide with Shelf or Bin OBBs.
    Uses circle-vs-OBB collision detection where SharedStuff objects are represented as circles
    using their KEEPOUT_RADIUS."""

    def __init__(self):
        super().__init__(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([np.inf])
        )

    def eval(self, scene_tree):
        """Evaluate minimum separation distance between all SharedStuff objects and all Shelves/Bins.
        Returns positive values if separated, negative if overlapping."""

        # Find both populated and empty container variants
        shelves = scene_tree.find_nodes_by_type(Shelf)
        bins = scene_tree.find_nodes_by_type(ClutteredBin)
        empty_shelves = scene_tree.find_nodes_by_type(EmptyShelf)
        empty_bins = scene_tree.find_nodes_by_type(EmptyClutteredBin)

        # Combine both types
        all_shelves = shelves + empty_shelves
        all_bins = bins + empty_bins

        # Collect all tabletop objects that need collision checking
        tabletop_objects = []
        for obj_type in TabletopObjectTypes:
            objs = scene_tree.find_nodes_by_type(obj_type)
            for obj in objs:
                if obj.observed:
                    tabletop_objects.append(obj)

        # If no tabletop objects or no shelves/bins, constraint is satisfied
        if len(tabletop_objects) == 0 or (len(all_shelves) == 0 and len(all_bins) == 0):
            return torch.tensor([[1.0]])

        separations = []

        # Check each tabletop object against all shelves (both populated and empty)
        for obj in tabletop_objects:
            obj_radius = obj.KEEPOUT_RADIUS

            for shelf in all_shelves:
                # Use EmptyShelf dimensions if it's an empty shelf, otherwise use Shelf dimensions
                if isinstance(shelf, EmptyShelf):
                    shelf_width = EmptyShelf.WIDTH / 2.0
                    shelf_length = EmptyShelf.LENGTH / 2.0
                else:
                    shelf_width = Shelf.WIDTH / 2.0
                    shelf_length = Shelf.LENGTH / 2.0

                separation = self._check_circle_obb_separation(
                    obj, obj_radius, shelf, shelf_width, shelf_length
                )
                separations.append(separation)

            # Check against all bins (both populated and empty)
            for bin_node in all_bins:
                # Use EmptyClutteredBin dimensions if it's an empty bin, otherwise use ClutteredBin dimensions
                if isinstance(bin_node, EmptyClutteredBin):
                    bin_half_x = EmptyClutteredBin.bin_dims[0] / 2.0
                    bin_half_y = EmptyClutteredBin.bin_dims[1] / 2.0
                else:
                    bin_half_x = ClutteredBin.bin_dims[0] / 2.0
                    bin_half_y = ClutteredBin.bin_dims[1] / 2.0

                separation = self._check_circle_obb_separation(
                    obj,
                    obj_radius,
                    bin_node,
                    bin_half_x,  # X half-extent
                    bin_half_y,  # Y half-extent
                )
                separations.append(separation)

        # Return minimum separation across all pairs
        return torch.stack(separations, axis=0).reshape(-1, 1)

    def _check_circle_obb_separation(
        self, circle_obj, circle_radius, obb_obj, obb_half_width, obb_half_length
    ):
        """Check separation between a circle and an oriented bounding box in 2D.

        Args:
            circle_obj: Node with translation representing circle center
            circle_radius: Radius of the circle
            obb_obj: Node with translation and rotation representing OBB
            obb_half_width: Half-width of OBB (X-axis extent)
            obb_half_length: Half-length of OBB (Y-axis extent)

        Returns:
            Separation distance (positive if separated, negative if overlapping)
        """
        # Extract 2D positions
        circle_pos = circle_obj.translation[:2]
        obb_pos = obb_obj.translation[:2]
        obb_rot = obb_obj.rotation[:2, :2]  # 2x2 XY rotation matrix

        # Transform circle center into OBB's local frame
        circle_pos_local = torch.matmul(obb_rot.T, circle_pos - obb_pos)

        # Find closest point on OBB to circle center (in OBB local frame)
        # Clamp circle position to OBB bounds
        closest_x = torch.clamp(circle_pos_local[0], -obb_half_width, obb_half_width)
        closest_y = torch.clamp(circle_pos_local[1], -obb_half_length, obb_half_length)
        closest_point_local = torch.tensor([closest_x, closest_y])

        # Compute distance from circle center to closest point on OBB
        dist_vector = circle_pos_local - closest_point_local
        dist = torch.norm(dist_vector)

        # Separation = distance - radius
        # Positive if separated, negative if overlapping
        separation = dist - circle_radius

        return separation

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()
