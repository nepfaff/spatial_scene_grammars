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
    (stacked_board_games | null) | (standing_board_games | null) | (speaker | null) |
    (book | null) | null

plate -> toast | null
stacked_board_games -> lying_board_game | null
standing_board_games -> standing_board_game | null

lying_board_game -> lying_board_game | null
standing_board_game -> standing_board_game | null

big_bowl -> null
bowl -> null
toast -> null
coke -> null
tea_bottle -> null
lamp -> null
speaker -> null
book -> null
large_board_game (possible multiple options) -> null

# NOTE: Could use box collision geometries for board games but might require mesh re-alignment.

# TODO: Book options from gazebo and manipulation
# TODO: Max board game stack height (see dumpling grammar)

# TODO: Next is finishing shelf_setting.

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

    # Shelf surface offsets from origin (z-coordinate).
    BOTTOM_OFFSET = -0.3915
    LOWER_OFFSET = -0.12315
    UPPER_OFFSET = 0.13915
    TOP_OFFSET = 0.4075

    def __init__(self, tf):
        # NOTE: Static shelf geometry is added at simulation time using a welded
        # directive.
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

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
                child_type=Null,
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


class LampOrNull(RepeatingSetNode):
    SAVE_RADIUS = 0.015
    KEEP_OUT_RADIUS = 0.02

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
        ]


class BigBowlOrNull(OrNode):
    SAVE_RADIUS = 0.11
    KEEP_OUT_RADIUS = 0.1

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
                child_type=BigBowl,
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
            rule_probs=torch.tensor([0.5, 0.5]),
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
    KEEP_OUT_RADIUS = (
        max(
            LyingBalderdashBoardGame.WIDTH,
            LyingBalderdashBoardGame.LENGTH,
        )
        / 2.0
    )
    SAVE_RADIUS = KEEP_OUT_RADIUS + 0.01

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
                child_type=LyingBoardGame,
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
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 500])
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
            rule_probs=torch.tensor([1.0, 0.0]),  # TODO: Adjust probs
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

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH)

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

    KEEP_OUT_RADIUS = max(WIDTH, LENGTH)

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


class BigBowl(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/bowls/generic_fruit_bowl.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)


class MinNumObjectsConstraint(StructureConstraint):
    def __init__(self, min_num_objects, table_node_type):
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
