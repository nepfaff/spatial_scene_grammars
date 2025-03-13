import torch
from pydrake.all import RigidTransform, RollPitchYaw

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *

""" 
Shelf (no geometry, static directive added after) -> top_shelf_setting_or_large_board_game &
    shelf_setting_or_large_board_game (upper) &
    shelf_setting_or_large_board_game (lower) &
    shelf_setting_or_large_board_game (bottom)
    
top_shelf_setting_or_large_board_game -> top_shelf_setting | large_board_game | null
shelf_setting_or_large_board_game -> shelf_setting | large_board_game | null

top_shelf_setting -> (lamp | null) | (big_bowl | null) | (stacked_board_games | null) |
    (speaker | null)
shelf_setting -> (bowl | null) | (plate | null) | (coke | null) | (tea_bottle | null) |
    (stacked_board_games | null) | (book | null) | (speaker | null) |
    (nintendo_game | null) | (stacking_ring | null) | (toy_train | null)

plate -> toast | pear | apple | null
stacked_board_games -> lying_board_game | null
standing_board_games -> standing_board_game | null

lying_board_game -> lying_board_game | null
standing_board_game -> standing_board_game | null

big_bowl -> null
toast -> null
pear -> null
apple -> null
coke -> null
tea_bottle -> null
lamp -> null
speaker -> null
book -> null
large_board_game (possible multiple options) -> null
nintendo_game -> nintendo_game | null
stacking_ring -> null
toy_train -> null
# NOTE: Could use box collision geometries for board games but might require mesh re-alignment.

# TODO: Bad collisions as rejection sampling constraint (keep objects out of object stacks as in table scenes)
# TODO: Add plate and toast. Toast should have max stack of 3.

Model visualizer is very useful:
```
python3 -m pydrake.visualization.model_visualizer \
~/robot_locomotion/spatial_scene_grammars/spatial_scene_grammars_examples/tri_living_room_shelf/gazebo/models/Balderdash_Game/lying_model.sdf
```
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


class Shelf(AndNode):
    # package://drake_models/manipulation_station/shelves.sdf

    # Dimensions in meters.
    WIDTH = 0.3  # x-coordinate
    LENGTH = 0.6  # y-coordinate
    HEIGHT = 0.783  # z-coordinate

    KEEPOUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    # Shelf surface offsets from origin (z-coordinate).
    BOTTOM_OFFSET = -0.3915
    LOWER_OFFSET = -0.12315
    UPPER_OFFSET = 0.13915
    TOP_OFFSET = 0.4075

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=True)
        geom_tf = torch.eye(4)
        geom.register_model_file(
            geom_tf, "package://drake_models/manipulation_station/shelves.sdf"
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=TopShelfSettingOrLargeBoardGame,
                xyz_rule=SamePositionRule(offset=torch.tensor([0, 0, cls.TOP_OFFSET])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ShelfSettingOrLargeBoardGame,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0, 0, cls.UPPER_OFFSET])
                ),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ShelfSettingOrLargeBoardGame,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0, 0, cls.LOWER_OFFSET])
                ),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ShelfSettingOrLargeBoardGame,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0, 0, cls.BOTTOM_OFFSET])
                ),
                rotation_rule=SameRotationRule(),
            ),
        ]


class TopShelfSettingOrLargeBoardGame(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.8, 0.05, 0.05, 0.1]),
            observed=False,
            physics_geometry_info=None,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=TopShelfSetting,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LargeBoardGame,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.0]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2.0)),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(  # Same as above but with flipped yaw.
                child_type=LargeBoardGame,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.0]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, -np.pi / 2.0)),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class TopShelfSetting(AndNode):

    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=LampOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=BigBowlOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=StackedBoardGamesOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=JBLSpeakerOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class ShelfSettingOrLargeBoardGame(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.7, 0.1, 0.1, 0.1]),
            observed=False,
            physics_geometry_info=None,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=ShelfSetting,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LargeBoardGame,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.01, 0.0, 0.0]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2.0)),
                    np.array([1e6, 1e6, 1000]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(  # Same as above but with flipped yaw.
                child_type=LargeBoardGame,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.01, 0.0, 0.0]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(0.0, 0.0, -np.pi / 2.0)),
                    np.array([1e6, 1e6, 1000]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class ShelfSetting(AndNode):

    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=StackedBoardGamesOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=JBLSpeakerOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=CokeCanOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=TeaBottleOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=StandingEatToLiveBookOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=NintendoGameOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=StackingRingOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ToyTrainOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=PlateOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class Plate(OrNode):
    KEEP_OUT_RADIUS = 0.11

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/plates/carlisle_plate_mesh_collision.sdf",
        )
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor(np.ones(4) / 4),
            physics_geometry_info=geom,
            observed=True,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Toast,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.021]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Apple,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.043]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Pear,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.048]),
                    variance=torch.tensor([0.001**2, 0.001**2, 1e-16]),
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


class PlateOrNull(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.8]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Plate,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Plate.KEEP_OUT_RADIUS,
                            -Shelf.LENGTH / 2 + Plate.KEEP_OUT_RADIUS,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Plate.KEEP_OUT_RADIUS,
                            Shelf.LENGTH / 2 - Plate.KEEP_OUT_RADIUS,
                            0.001,
                        ]
                    ),
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


class Lamp(TerminalNode):
    SAVE_RADIUS = 0.015
    KEEP_OUT_RADIUS = 0.02

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/3D_Dollhouse_Lamp/model.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class LampOrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.8, max_children=2, start_at_one=False
            ),  # No lamp with probability of ~0.8
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Lamp,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Lamp.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Lamp.SAVE_RADIUS,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Lamp.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Lamp.SAVE_RADIUS,
                            0.001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]


class BigBowl(TerminalNode):
    SAVE_RADIUS = 0.11
    KEEP_OUT_RADIUS = 0.1

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/bowls/generic_fruit_bowl.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class BigBowlOrNull(OrNode):

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.8]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=BigBowl,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + BigBowl.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + BigBowl.SAVE_RADIUS,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - BigBowl.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - BigBowl.SAVE_RADIUS,
                            0.001,
                        ]
                    ),
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


class StandingEatToLiveBook(TerminalNode):
    WIDTH = 0.14  # x-coordinate
    LENGTH = 0.05  # y-coordinate
    HEIGHT = 0.21  # z-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book/model.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class StandingEatToLiveBookOrNull(OrNode):
    SAVE_RADIUS = max(StandingEatToLiveBook.WIDTH, StandingEatToLiveBook.LENGTH)
    KEEP_OUT_RADIUS = SAVE_RADIUS + 0.01

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.8]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            # Leaning against left wall.
            ProductionRule(
                child_type=StandingEatToLiveBook,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + StandingEatToLiveBook.WIDTH / 2 + 0.01,
                            -Shelf.LENGTH / 2 + StandingEatToLiveBook.LENGTH / 2,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - StandingEatToLiveBook.WIDTH / 2 - 0.01,
                            -Shelf.LENGTH / 2
                            + StandingEatToLiveBook.LENGTH / 2
                            + 0.001,
                            0.001,
                        ]
                    ),
                ),
                rotation_rule=SameRotationRule(
                    offset=torch.from_numpy(
                        RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi)).matrix()
                    )
                ),
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class StackingRing(TerminalNode):
    KEEP_OUT_RADIUS = 0.04

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/STACKING_RING/model.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class StackingRingOrNull(OrNode):
    SAVE_RADIUS = StackingRing.KEEP_OUT_RADIUS
    KEEP_OUT_RADIUS = StackingRing.KEEP_OUT_RADIUS

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.8]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=StackingRing,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + StackingRing.KEEP_OUT_RADIUS,
                            -Shelf.LENGTH / 2 + StackingRing.KEEP_OUT_RADIUS,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - StackingRing.KEEP_OUT_RADIUS,
                            Shelf.LENGTH / 2 - StackingRing.KEEP_OUT_RADIUS,
                            0.001,
                        ]
                    ),
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


class ToyTrain(TerminalNode):
    WIDTH = 0.04  # x-coordinate
    LENGTH = 0.09  # y-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/Thomas_Friends_Wooden_Railway_Talking_Thomas/model.sdf",
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True,
        )


class ToyTrainOrNull(OrNode):
    SAVE_RADIUS = max(ToyTrain.WIDTH, ToyTrain.LENGTH)
    KEEP_OUT_RADIUS = SAVE_RADIUS + 0.01

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.8]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=ToyTrain,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + ToyTrain.KEEP_OUT_RADIUS,
                            -Shelf.LENGTH / 2 + ToyTrain.KEEP_OUT_RADIUS,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - ToyTrain.KEEP_OUT_RADIUS,
                            Shelf.LENGTH / 2 - ToyTrain.KEEP_OUT_RADIUS,
                            0.001,
                        ]
                    ),
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


class NintendoGame(OrNode):
    WIDTH = 0.14  # x-coordinate
    LENGTH = 0.125  # y-coordinate
    HEIGHT = 0.0125  # z-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/Pokmon_Y_Nintendo_3DS_Game/model.sdf",
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True,
            rule_probs=torch.tensor([0.3, 0.7]),
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=NintendoGame,
                xyz_rule=WorldFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, NintendoGame.HEIGHT]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
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


class NintendoGameOrNull(RepeatingSetNode):
    SAVE_RADIUS = max(NintendoGame.WIDTH, NintendoGame.LENGTH) / 2.0
    KEEP_OUT_RADIUS = SAVE_RADIUS + 0.01

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.7, max_children=3, start_at_one=False
            ),  # No game can with probability of ~0.7
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=NintendoGame,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + NintendoGame.WIDTH,
                            -Shelf.LENGTH / 2 + NintendoGame.LENGTH,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - NintendoGame.WIDTH,
                            Shelf.LENGTH / 2 - NintendoGame.LENGTH,
                            0.001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class CokeCan(TerminalNode):
    KEEP_OUT_RADIUS = 0.0334
    SAVE_RADIUS = KEEP_OUT_RADIUS + 0.01

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/junk/coke.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class CokeCanOrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.7, max_children=6, start_at_one=False
            ),  # No coke can with probability of ~0.7
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=CokeCan,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + CokeCan.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + CokeCan.SAVE_RADIUS,
                            0.063,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - CokeCan.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - CokeCan.SAVE_RADIUS,
                            0.064,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class TeaBottle(TerminalNode):
    KEEP_OUT_RADIUS = 0.0325
    SAVE_RADIUS = KEEP_OUT_RADIUS + 0.01

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/junk/tea_bottle.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class TeaBottleOrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.7, max_children=6, start_at_one=False
            ),  # No tea bottle with probability of ~0.7
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=TeaBottle,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + TeaBottle.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + TeaBottle.SAVE_RADIUS,
                            0.104,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - TeaBottle.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - TeaBottle.SAVE_RADIUS,
                            0.105,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class JBLSpeaker(TerminalNode):
    WIDTH = 0.175  # x-coordinate
    LENGTH = 0.07  # y-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/JBL_Charge_Speaker_portable_wireless_wired_Green/model.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class JBLSpeakerOrNull(OrNode):
    SAVE_RADIUS = JBLSpeaker.KEEP_OUT_RADIUS
    KEEP_OUT_RADIUS = JBLSpeaker.KEEP_OUT_RADIUS

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.1, 0.9]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=JBLSpeaker,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + cls.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + cls.SAVE_RADIUS,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - cls.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - cls.SAVE_RADIUS,
                            0.001,
                        ]
                    ),
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


class LyingBalderdashBoardGame(OrNode):
    WIDTH = 0.2  # x-coordinate
    LENGTH = 0.264  # y-coordinate
    HEIGHT = 0.06  # z-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/Balderdash_Game/lying_model.sdf",
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True,
            rule_probs=torch.tensor([0.3, 0.7]),
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=LyingBoardGame,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0.0, 0.0, LyingBalderdashBoardGame.HEIGHT]),
                ),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class StackedBoardGamesOrNull(OrNode):
    MAX_WIDTH = LyingBalderdashBoardGame.WIDTH
    MAX_LENGTH = LyingBalderdashBoardGame.LENGTH
    KEEP_OUT_RADIUS = max(MAX_WIDTH, MAX_LENGTH) / 2.0
    SAVE_RADIUS = KEEP_OUT_RADIUS + 0.01

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.15, 0.15, 0.7]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=LyingBoardGame,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + cls.MAX_WIDTH / 2 + 0.03,
                            -Shelf.LENGTH / 2 + cls.MAX_LENGTH / 2,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - cls.MAX_WIDTH / 2,
                            Shelf.LENGTH / 2 - cls.MAX_LENGTH / 2,
                            0.001,
                        ]
                    ),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 500])
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(  # Rotated by 90 degrees about yaw axis
                child_type=LyingBoardGame,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + cls.MAX_LENGTH / 2 + 0.03,
                            -Shelf.LENGTH / 2 + cls.MAX_WIDTH / 2,
                            0.0,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - cls.MAX_LENGTH / 2,
                            Shelf.LENGTH / 2 - cls.MAX_WIDTH / 2,
                            0.001,
                        ]
                    ),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(RollPitchYaw(roll=0.0, pitch=0.0, yaw=np.pi / 2)),
                    np.array([1e6, 1e6, 500]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class LyingBoardGame(OrNode):

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor(
                [1.0, 0.0]
            ),  # TODO: Adjust probs once have multiple small board games
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=LyingBalderdashBoardGame,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.0]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 500])
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=LyingClueBoardGame,  # TODO: replace with smaller game
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.0]),
                    variance=torch.tensor([0.01**2, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 500])
                ),  # Allow some yaw rotation.
            ),
        ]
        return rules


class LargeBoardGame(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            physics_geometry_info=None,
            observed=False,
            rule_probs=torch.tensor([1 / 3, 1 / 3, 1 / 3]),
        )

    @classmethod
    def generate_rules(cls):
        # The parent specifies the position and rotation rules.
        rules = [
            ProductionRule(
                child_type=LyingClueBoardGame,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LyingMonopolyBoardGame,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LyingSlidersBoardGame,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class LyingClueBoardGame(OrNode):
    WIDTH = 0.5  # x-coordinate
    LENGTH = 0.25  # y-coordinate
    HEIGHT = 0.055  # z-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/Clue_Board_Game_Classic_Edition/lying_model.sdf",
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True,
            rule_probs=torch.tensor([0.3, 0.7]),
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=LargeBoardGame,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0.0, 0.0, LyingClueBoardGame.HEIGHT]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 5000])
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class LyingMonopolyBoardGame(OrNode):
    WIDTH = 0.4  # x-coordinate
    LENGTH = 0.265  # y-coordinate
    HEIGHT = 0.055  # z-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/My_Monopoly_Board_Game/lying_model.sdf",
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True,
            rule_probs=torch.tensor([0.3, 0.7]),
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=LargeBoardGame,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0.0, 0.0, LyingMonopolyBoardGame.HEIGHT]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 5000])
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class LyingSlidersBoardGame(OrNode):
    WIDTH = 0.4  # x-coordinate
    LENGTH = 0.265  # y-coordinate
    HEIGHT = 0.069  # z-coordinate

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH) / 2.0

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://gazebo/models/Sorry_Sliders_Board_Game/lying_model.sdf",
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True,
            rule_probs=torch.tensor([0.3, 0.7]),
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=LargeBoardGame,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0.0, 0.0, LyingSlidersBoardGame.HEIGHT]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 5000])
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class Toast(TerminalNode):
    KEEP_OUT_RADIUS = 0.01  # Not measured

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/food/sandwich/fake_toasted_bread_slice_mesh_collision.sdf",
        )
        super().__init__(
            tf=tf,
            physics_geometry_info=geom,
            observed=True,
        )


class Apple(TerminalNode):
    KEEP_OUT_RADIUS = 0.01  # Not measured

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/apples/gala_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Pear(TerminalNode):
    KEEP_OUT_RADIUS = 0.01  # Not measured

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/pears/bose_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)


class MinNumObjectsConstraint(StructureConstraint):
    def __init__(self, min_num_objects, table_node_type=Shelf):
        super().__init__(
            lower_bound=torch.tensor([min_num_objects]),
            upper_bound=torch.tensor([torch.inf]),
        )
        self.table_node_type = table_node_type

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(self.table_node_type)
        num_objects = 0
        for table in tables:
            # Collect table children xyz poses in table frame
            objs = scene_tree.get_children_recursive(table)
            observed_objs = [obj for obj in objs if obj.observed]
            num_objects += len(observed_objs)
        return torch.tensor([num_objects])

    def add_to_ik_prog(
        self, scene_tree, ik, mbp, mbp_context, node_to_free_body_ids_map
    ):
        raise NotImplementedError()


class BoardGameStackHeightConstraint(StructureConstraint):
    # The largest stack of board games should be less than N games tall
    def __init__(self, max_height=5):
        lb = torch.tensor([0])
        ub = torch.tensor([max_height])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        board_games = scene_tree.find_nodes_by_type(LyingBoardGame)
        tallest_stack = 0
        # For each board game, count how many ancestors are in the
        # board game stack pattern (LyingBoardGame -> specific game -> LyingBoardGame...)
        for game in board_games:
            current_node = game
            stack = 0
            while current_node is not None:
                if isinstance(current_node, LyingBoardGame):
                    stack += 1
                current_node = scene_tree.get_parent(current_node)
            tallest_stack = max(tallest_stack, stack)
        return torch.tensor([tallest_stack])


class LargeBoardGameStackHeightConstraint(StructureConstraint):
    # The largest stack of large board games should be less than N games tall
    def __init__(self, max_height=3):
        lb = torch.tensor([0])
        ub = torch.tensor([max_height])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        board_games = scene_tree.find_nodes_by_type(LargeBoardGame)
        tallest_stack = 0
        # For each board game, count how many ancestors are in the
        # board game stack pattern (LargeBoardGame -> specific game -> LargeBoardGame...)
        for game in board_games:
            current_node = game
            stack = 0
            while current_node is not None:
                if isinstance(current_node, LargeBoardGame):
                    stack += 1
                current_node = scene_tree.get_parent(current_node)
            tallest_stack = max(tallest_stack, stack)
        return torch.tensor([tallest_stack])


def compute_objects_not_in_collision_with_stacks(scene_tree):
    shelves = scene_tree.find_nodes_by_type(Shelf)
    min_distances = []

    for shelf in shelves:
        # Get all shelf layer nodes.
        shelf_layers = scene_tree.get_children_recursive(
            shelf, (TopShelfSettingOrLargeBoardGame, ShelfSettingOrLargeBoardGame)
        )

        # Process each shelf layer separately.
        # Objects from different layers can't collide.
        for layer in shelf_layers:
            # These will already be considered by the base stack object.
            exclude_object_classes = (Apple, Pear, Toast)

            # Get all objects in this layer.
            layer_objects = [
                obj
                for obj in scene_tree.get_children_recursive(layer)
                if obj.observed and not isinstance(obj, exclude_object_classes)
            ]

            if not layer_objects:
                continue

            # Get stack objects in this layer.
            object_stack_classes = (LyingBalderdashBoardGame, NintendoGame, Plate)
            stack_objects = [
                obj for obj in layer_objects if isinstance(obj, object_stack_classes)
            ]

            if not stack_objects:
                continue

            # Extract translations and keep out radii.
            stack_translations = torch.stack(
                [stack_obj.translation[:2] for stack_obj in stack_objects]
            )
            layer_translations = torch.stack(
                [obj.translation[:2] for obj in layer_objects]
            )
            stack_keep_out_radii = torch.tensor(
                [stack_obj.KEEP_OUT_RADIUS for stack_obj in stack_objects]
            )
            layer_keep_out_radii = torch.tensor(
                [obj.KEEP_OUT_RADIUS for obj in layer_objects]
            )

            # Compute distances within this layer.
            distances = torch.cdist(stack_translations, layer_translations)

            # Create a mask to ignore self-distances.
            mask = torch.ones_like(distances, dtype=torch.bool)
            for i, stack_obj in enumerate(stack_objects):
                # Find the index of this stack object in layer_objects
                obj_idx = layer_objects.index(stack_obj)
                # Mask out the self-distance
                mask[i, obj_idx] = False

            # Apply the mask by setting masked distances to a large value.
            large_value = 1000.0  # Should be larger than any realistic distance
            masked_distances = torch.where(mask, distances, large_value)

            # Calculate minimum distances for this layer.
            layer_min_distances = masked_distances - (
                stack_keep_out_radii.unsqueeze(1) + layer_keep_out_radii.unsqueeze(0)
            ).expand(distances.shape)
            min_distances.append(torch.flatten(layer_min_distances))

    if not min_distances:
        # If no objects to check, constraint is trivially satisfied.
        return torch.tensor([1.0])

    # Combine all minimum distances and return.
    return torch.cat(min_distances).unsqueeze(1)


class ObjectsNotInCollisionWithStacksConstraintPose(PoseConstraint):
    def __init__(self):
        super().__init__(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([torch.inf])
        )

    def eval(self, scene_tree):
        return compute_objects_not_in_collision_with_stacks(scene_tree)


class ObjectsNotInCollisionWithStacksConstraintStructure(StructureConstraint):
    def __init__(self):
        super().__init__(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([torch.inf])
        )

    def eval(self, scene_tree):
        return compute_objects_not_in_collision_with_stacks(scene_tree)
