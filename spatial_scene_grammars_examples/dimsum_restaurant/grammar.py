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
            rule_probs=torch.tensor([0.3, 0.4, 0.3]),
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
        # Create a custom distribution that starts at 3 tables
        rule_probs = torch.zeros(11)  # For 0 to 10 tables
        rule_probs[0:3] = 0.0  # Zero probability for 0, 1, or 2 tables

        # Geometric distribution starting at 3
        p = 0.3  # Lower value = higher probability of more tables
        for k in range(3, 11):
            rule_probs[k] = (1 - p) ** (k - 3) * p

        # Normalize to sum to 1
        rule_probs = rule_probs / torch.sum(rule_probs)

        super().__init__(
            tf=tf, physics_geometry_info=None, observed=False, rule_probs=rule_probs
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Table,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    torch.tensor([-2.5, -3.0, 0.0]), torch.tensor([3.0, 3.0, 0.0])
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
                p=0.2, max_children=5, start_at_one=False
            ),
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Shelf,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    torch.tensor([-4.7501, -3.0, -0.3881]),
                    torch.tensor([-4.75, 3.0, -0.388]),
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
                continue

            # Vectorize computation
            xys = torch.stack([obj.translation[:2] for obj in objs], axis=0)
            keepout_dists = torch.tensor([obj.KEEPOUT_RADIUS for obj in objs])

            # Compute all pairwise distances at once
            dists = torch.cdist(xys, xys)

            # Create matrices of radii for each pair
            keepout_matrix1 = keepout_dists.unsqueeze(1).expand(-1, len(keepout_dists))
            keepout_matrix2 = keepout_dists.unsqueeze(0).expand(len(keepout_dists), -1)
            keepout_sum = keepout_matrix1 + keepout_matrix2

            # Calculate margins
            margins = dists - keepout_sum

            # Get only lower triangular non-diagonal elements
            N = xys.shape[0]
            rows, cols = torch.tril_indices(N, N, -1)
            all_dists.append(margins[rows, cols].reshape(-1, 1))

        if len(all_dists) > 0:
            return torch.cat(all_dists, axis=0)
        else:
            return torch.empty(size=(0, 1))

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class TallStackConstraint(StructureConstraint):
    # The largest stack of steamers is at least 3 steamers tall.
    def __init__(self):
        lb = torch.tensor([3.0])
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


class TablesChairsAndShelvesNotInCollisionConstraint(StructureConstraint):
    # Ensures that tables, chairs, and shelves don't collide with each other
    def __init__(self, debug=False):
        # Constraint is satisfied when all distances are positive
        self.debug = debug
        super().__init__(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([torch.inf])
        )

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
        shelves = scene_tree.find_nodes_by_type(Shelf)

        # If there are no objects to check, constraint is trivially satisfied
        if len(tables) + len(shelves) <= 1:
            if self.debug:
                print("No tables/shelves to check for collisions")
            return torch.tensor([1.0])

        # Collect all objects with positions, radii, and parent info
        objects = []
        table_to_index = {}  # Map tables to their index in objects list

        for i, table in enumerate(tables):
            table_idx = len(objects)
            table_to_index[table] = table_idx
            objects.append((table.translation[:2], table.KEEPOUT_RADIUS, "table", None))

            # Get chairs associated with this table
            table_chairs = [
                obj
                for obj in scene_tree.get_children_recursive(table)
                if isinstance(obj, Chair) and obj.observed
            ]

            for chair in table_chairs:
                objects.append(
                    (chair.translation[:2], chair.KEEPOUT_RADIUS, "chair", table_idx)
                )

        for shelf in shelves:
            objects.append((shelf.translation[:2], shelf.KEEPOUT_RADIUS, "shelf", None))

        # If no objects, return trivially satisfied constraint
        if len(objects) <= 1:
            if self.debug:
                print("Only one object to check")
            return torch.tensor([1.0])

        # Create tensors for vectorized computation
        positions = torch.stack([obj[0] for obj in objects])
        radii = torch.tensor([obj[1] for obj in objects])
        types = [obj[2] for obj in objects]
        parent_indices = [obj[3] for obj in objects]

        # Compute all pairwise distances at once
        distances = torch.cdist(positions, positions)

        # Create matrices of radii for each pair
        radii_matrix1 = radii.unsqueeze(1).expand(-1, len(radii))
        radii_matrix2 = radii.unsqueeze(0).expand(len(radii), -1)

        # Calculate margins
        margins = distances - (radii_matrix1 + radii_matrix2)

        # Get only lower triangular non-diagonal elements (to avoid self-comparisons)
        n = len(objects)
        rows, cols = torch.tril_indices(n, n, -1)

        # Filter out chair-table pairs where the chair belongs to that table
        valid_pairs = []
        for i, j in zip(rows.tolist(), cols.tolist()):
            # Skip chair-table comparisons where chair belongs to the table
            if (
                types[i] == "chair" and types[j] == "table" and parent_indices[i] == j
            ) or (
                types[j] == "chair" and types[i] == "table" and parent_indices[j] == i
            ):
                continue

            # Skip chair-chair comparisons from the same table
            if (
                types[i] == "chair"
                and types[j] == "chair"
                and parent_indices[i] == parent_indices[j]
            ):
                continue

            valid_pairs.append((i, j))

        if not valid_pairs:
            if self.debug:
                print("No valid pairs to check")
            return torch.tensor([1.0])

        valid_rows = torch.tensor([p[0] for p in valid_pairs])
        valid_cols = torch.tensor([p[1] for p in valid_pairs])

        result = margins[valid_rows, valid_cols].reshape(-1, 1)

        if self.debug:
            print(f"Collision constraint: {result.shape[0]} pairs checked")
            if result.shape[0] > 0:
                print(f"  Min margin: {result.min().item():.4f}")
                if (result < 0).any():
                    num_violations = (result < 0).sum().item()
                    print(f"  {num_violations} collision(s) detected")

                    # Print details about the collisions
                    for idx in range(result.shape[0]):
                        if result[idx] < 0:
                            i, j = valid_rows[idx].item(), valid_cols[idx].item()
                            print(
                                f"  Collision between {types[i]} and {types[j]}: "
                                f"margin={result[idx].item():.4f}"
                            )

        return result

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()
