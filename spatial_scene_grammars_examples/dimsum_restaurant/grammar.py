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

# TODO: Need to add shelves
# TODO: Add structure constraint for preventing collision between tables and shelves
# TODO: Update greg_table on Anzu
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


class PersonalPlate(TerminalNode):
    KEEPOUT_RADIUS = 0.14

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg_table/models/plates_cups_and_bowls/plates/Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring/model_simplified.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Teacup(TerminalNode):
    KEEPOUT_RADIUS = 0.07

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg_table/models/plates_cups_and_bowls/cups/coffee_cup_white/model_simplified.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Teapot(TerminalNode):
    KEEPOUT_RADIUS = 0.1

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg_table/models/plates_cups_and_bowls/cups/Threshold_Porcelain_Teapot_White/model_simplified.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class ServingDish(TerminalNode):
    KEEPOUT_RADIUS = 0.2

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg_table/models/plates_cups_and_bowls/plates/Threshold_Dinner_Plate_Square_Rim_White_Porcelain/model_simplified.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)


class SteamerBottom(OrNode):
    KEEPOUT_RADIUS = 0.12

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg_table/models/misc/steamer_bottom/model.sdf",
        )
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.25, 0.4, 0.35]),
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
            "package://greg_table/models/misc/steamer_top/model.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


TabletopObjectTypes = (
    PersonalPlate,
    Teacup,
    Teapot,
    ServingDish,
    SteamerBottom,
    SteamerTop,
)


class Chair(TerminalNode):
    KEEPOUT_RADIUS = 0.35

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://greg_table/models/misc/chair/model.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class PersonalPlateAndTeacup(AndNode):
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=PersonalPlate,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.001, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Teacup,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.25, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.005, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class PlaceSetting(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.5, 0.05, 0.05, 0.4]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=PersonalPlateAndTeacup,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=PersonalPlate,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.001, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Teacup,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.25, 0.0, 0.00]),
                    variance=torch.tensor([0.001, 0.005, 1e-16]),
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


class PlaceSettings(AndNode):
    DISTANCE_FROM_CENTER = 0.5
    CHAIR_DISTANCE_FROM_CENTER = 1.1
    TABLE_HEIGHT = 0.785
    CHAIR_YAW_VARIANCE = 50

    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([-cls.DISTANCE_FROM_CENTER, 0.0, 0.0])
                ),
                rotation_rule=SameRotationRule(
                    offset=torch.tensor(
                        RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)).matrix()
                    )
                ),
            ),
            ProductionRule(
                child_type=Chair,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor(
                        [-cls.CHAIR_DISTANCE_FROM_CENTER, 0.0, -cls.TABLE_HEIGHT]
                    ),
                    variance=torch.tensor([0.005, 0.03, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, -np.pi / 2)),
                    np.array([1e6, 1e6, cls.CHAIR_YAW_VARIANCE]),
                ),
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([cls.DISTANCE_FROM_CENTER, 0.0, 0.0]),
                ),
                rotation_rule=SameRotationRule(
                    offset=torch.tensor(
                        RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi)).matrix()
                    )
                ),
            ),
            ProductionRule(
                child_type=Chair,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor(
                        [cls.CHAIR_DISTANCE_FROM_CENTER, 0.0, -cls.TABLE_HEIGHT]
                    ),
                    variance=torch.tensor([0.005, 0.03, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2)),
                    np.array([1e6, 1e6, cls.CHAIR_YAW_VARIANCE]),
                ),
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0.0, cls.DISTANCE_FROM_CENTER, 0.0])
                ),
                rotation_rule=SameRotationRule(
                    offset=torch.tensor(
                        RotationMatrix(RollPitchYaw(0.0, 0.0, -np.pi / 2.0)).matrix()
                    )
                ),
            ),
            ProductionRule(
                child_type=Chair,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor(
                        [0.0, cls.CHAIR_DISTANCE_FROM_CENTER, -cls.TABLE_HEIGHT]
                    ),
                    variance=torch.tensor([0.005, 0.03, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi)),
                    np.array([1e6, 1e6, cls.CHAIR_YAW_VARIANCE]),
                ),
            ),
            ProductionRule(
                child_type=PlaceSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0.0, -cls.DISTANCE_FROM_CENTER, 0.0])
                ),
                rotation_rule=SameRotationRule(
                    offset=torch.tensor(
                        RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2.0)).matrix()
                    )
                ),
            ),
            ProductionRule(
                child_type=Chair,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor(
                        [0.0, -cls.CHAIR_DISTANCE_FROM_CENTER, -cls.TABLE_HEIGHT]
                    ),
                    variance=torch.tensor([0.005, 0.03, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),
                    np.array([1e6, 1e6, cls.CHAIR_YAW_VARIANCE]),
                ),
            ),
        ]


class SharedDishes(RepeatingSetNode):
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
                child_type=ServingDish,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.02, 0.02, 1e-16]),
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
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.01, 0.01, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedSteamers(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=4, start_at_one=True
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SteamerBottom,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.00]),
                    variance=torch.tensor([0.025, 0.025, 1e-16]),
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
                child_type=SharedDishes,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedSteamers,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class Table(AndNode):
    WIDTH = 1.25
    KEEPOUT_RADIUS = 0.8

    # Place settings + misc common dishware
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(
            geom_tf, "package://greg_table/models/misc/cafe_table/model.sdf"
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlaceSettings,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.0, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedStuff,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.0, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
        ]


class Tables(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.25, max_children=10, start_at_one=True
            ),
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Table,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    torch.tensor([-3.0, -3.0, 0.0]), torch.tensor([3.0, 3.0, 0.0])
                ),
                rotation_rule=UniformBoundedRevoluteJointRule(
                    axis=torch.tensor([0.0, 0.0, 1.0]),
                    center=torch.tensor([0.0]),
                    width=torch.tensor([2.0 * np.pi]),
                ),
            )
        ]


class Shelves(RepeatingSetNode):
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
                child_type=Shelf,
                # TODO: Need to adjust such that only get spawned at walls
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    torch.tensor([-3.0, -3.0, 0.0]), torch.tensor([3.0, 3.0, 0.0])
                ),
                rotation_rule=SameRotationRule(),
            )
        ]


class Restaurant(AndNode):
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


# Corresponding constraint set for the grammar.
class ObjectsOnTableConstraint(PoseConstraint):
    def __init__(self):
        lb = torch.tensor([-Table.WIDTH / 2.0 + 0.15, -Table.WIDTH / 2.0 + 0.15, -0.02])
        ub = torch.tensor([Table.WIDTH / 2.0 - 0.15, Table.WIDTH / 2.0 - 0.15, 1.0])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
        xyzs = []  # in parent table frame
        for table in tables:
            # Collect table children xyz poses in table frame
            objs = [
                node
                for node in scene_tree.get_children_recursive(table)
                if isinstance(node, TabletopObjectTypes) and not isinstance(node, Chair)
            ]
            for obj in objs:
                offset = torch.matmul(
                    table.rotation.T, obj.translation - table.translation
                )
                xyzs.append(offset)
        if len(xyzs) > 0:
            return torch.stack(xyzs, axis=0)
        else:
            return torch.empty(size=(0, 3))

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class ObjectOnTableSpacingConstraint(PoseConstraint):
    # Objects all a minimum distance apart on tabletop
    def __init__(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([np.inf])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
        all_dists = []
        for table in tables:
            objs = [
                node
                for node in scene_tree.get_children_recursive(table)
                if isinstance(node, TabletopObjectTypes)
                and not isinstance(
                    scene_tree.get_parent(node), SteamerBottom
                )  # Want stacked steamers to touch each other
                and not isinstance(node, Chair)
            ]
            if len(objs) <= 1:
                # print("no objects")
                continue
            xys = torch.stack([obj.translation[:2] for obj in objs], axis=0)
            keepout_dists = torch.tensor([obj.KEEPOUT_RADIUS for obj in objs])
            N = xys.shape[0]
            xys_rowwise = xys.unsqueeze(1).expand(-1, N, -1)
            keepout_dists_rowwise = keepout_dists.unsqueeze(1).expand(-1, N)
            xys_colwise = xys.unsqueeze(0).expand(N, -1, -1)
            keepout_dists_colwise = keepout_dists.unsqueeze(0).expand(N, -1)
            dists = (xys_rowwise - xys_colwise).square().sum(axis=-1)
            keepout_dists = keepout_dists_rowwise + keepout_dists_colwise

            # Get only lower triangular non-diagonal elems
            rows, cols = torch.tril_indices(N, N, -1)
            # Make sure pairwise dists > keepout dists
            dists = (dists - keepout_dists.square())[rows, cols].reshape(-1, 1)
            all_dists.append(dists)
        if len(all_dists) > 0:
            return torch.cat(all_dists, axis=0)
        else:
            return torch.empty(size=(0, 1))

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class TallStackConstraint(StructureConstraint):
    # The largest stack of steamers is at least 4 steamers tall.
    def __init__(self):
        lb = torch.tensor([4.0])
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


class NumStacksConstraint(StructureConstraint):
    # Each table must have at least 2 stacks.
    MIN_STACKS = 2

    def __init__(self):
        lb = torch.tensor(
            [0.0]
        )  # Lower bound is 0 since we'll check each table individually
        ub = torch.tensor([np.inf])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        # Find all tables in the restaurant
        tables = scene_tree.find_nodes_by_type(Table)

        # If there are no tables, return a failing constraint value
        if len(tables) == 0:
            return torch.tensor([-1.0])

        # Check each table for sufficient stacks
        all_margins = []

        for table in tables:
            # Find SharedSteamers nodes for this specific table
            shared_steamers_nodes = [
                node
                for node in scene_tree.get_children_recursive(table)
                if isinstance(node, SharedSteamers)
            ]

            # Count stacks for this table
            table_stacks = 0
            for shared_steamers in shared_steamers_nodes:
                table_stacks += len(list(scene_tree.successors(shared_steamers)))

            # Calculate how many stacks above/below the required minimum
            # Positive value means constraint is satisfied
            margin = table_stacks - self.MIN_STACKS
            all_margins.append(margin)

        # Return all margins as a tensor
        return torch.tensor(all_margins).reshape(-1, 1)

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class TablesAndChairsNotInCollisionConstraint(StructureConstraint):
    # Ensures that tables and chairs from different table groups don't collide
    def __init__(self):
        # Constraint is satisfied when all distances are positive
        super().__init__(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([torch.inf])
        )

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)

        # If there's only one table or no tables, constraint is trivially satisfied
        if len(tables) <= 1:
            return torch.tensor([1.0])

        all_margins = []

        # Compare each table with every other table
        for i, table1 in enumerate(tables):
            # Get table position and keepout radius
            table1_pos = table1.translation[:2]
            table1_radius = table1.KEEPOUT_RADIUS

            # Get chairs associated with this table
            table1_chairs = [
                obj
                for obj in scene_tree.get_children_recursive(table1)
                if isinstance(obj, Chair) and obj.observed
            ]

            # Extract chair positions and radii if there are any
            if table1_chairs:
                chair1_positions = torch.stack(
                    [chair.translation[:2] for chair in table1_chairs]
                )
                chair1_radii = torch.tensor(
                    [chair.KEEPOUT_RADIUS for chair in table1_chairs]
                )

            # Compare with all other tables
            for j in range(i + 1, len(tables)):
                table2 = tables[j]
                table2_pos = table2.translation[:2]
                table2_radius = table2.KEEPOUT_RADIUS

                # Table-to-table distance check
                table_distance = torch.norm(table1_pos - table2_pos)
                table_margin = table_distance - (table1_radius + table2_radius)
                all_margins.append(table_margin.reshape(1))

                # Get chairs associated with the second table
                table2_chairs = [
                    obj
                    for obj in scene_tree.get_children_recursive(table2)
                    if isinstance(obj, Chair) and obj.observed
                ]

                if table2_chairs:
                    chair2_positions = torch.stack(
                        [chair.translation[:2] for chair in table2_chairs]
                    )
                    chair2_radii = torch.tensor(
                        [chair.KEEPOUT_RADIUS for chair in table2_chairs]
                    )

                # Check chair-to-chair distances
                if table1_chairs and table2_chairs:
                    chair_distances = torch.cdist(chair1_positions, chair2_positions)
                    # Create matrices of radii for each pair
                    chair1_radii_matrix = chair1_radii.unsqueeze(1).expand(
                        -1, len(chair2_radii)
                    )
                    chair2_radii_matrix = chair2_radii.unsqueeze(0).expand(
                        len(chair1_radii), -1
                    )
                    # Calculate margins
                    chair_margins = chair_distances - (
                        chair1_radii_matrix + chair2_radii_matrix
                    )
                    all_margins.append(chair_margins.flatten())

                # Check table1 to chair2 distances
                if table2_chairs:
                    t1_c2_distances = torch.norm(
                        chair2_positions - table1_pos.unsqueeze(0), dim=1
                    )
                    t1_c2_margins = t1_c2_distances - (table1_radius + chair2_radii)
                    all_margins.append(t1_c2_margins)

                # Check table2 to chair1 distances
                if table1_chairs:
                    t2_c1_distances = torch.norm(
                        chair1_positions - table2_pos.unsqueeze(0), dim=1
                    )
                    t2_c1_margins = t2_c1_distances - (table2_radius + chair1_radii)
                    all_margins.append(t2_c1_margins)

        # If we have any margins to check, return them all
        if all_margins:
            return torch.cat(all_margins).unsqueeze(1)
        else:
            # If no comparisons were made, constraint is trivially satisfied
            return torch.tensor([1.0])

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class MinimumTablesConstraint(StructureConstraint):
    # Ensures that the restaurant has at least a minimum number of tables
    def __init__(self, min_tables=3):
        super().__init__(
            lower_bound=torch.tensor([float(min_tables)]),
            upper_bound=torch.tensor([torch.inf]),
        )
        self.min_tables = min_tables

    def eval(self, scene_tree):
        # Find all tables in the scene
        tables = scene_tree.find_nodes_by_type(Table)

        # Return the count of tables as a tensor
        return torch.tensor([float(len(tables))])

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()
