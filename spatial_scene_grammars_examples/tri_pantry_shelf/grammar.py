import torch
from pydrake.all import RigidTransform, RollPitchYaw

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *

""" 
Shelf (no geometry, static directive added after) -> top_shelf_setting &
    shelf_setting (upper) & shelf_setting (lower) & shelf_setting (bottom)
    
top_shelf_setting -> (big_bowl | null) & shelf_setting
shelf_setting -> (coke | null) & (tea_bottle | null) & (apple | null) & (pear | null) &
                    (avocado | null)

big_bowl -> null
pear -> null
apple -> null
coke -> null
tea_bottle -> null
avocado -> null

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
                child_type=TopShelfSetting,
                xyz_rule=SamePositionRule(offset=torch.tensor([0, 0, cls.TOP_OFFSET])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ShelfSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0, 0, cls.UPPER_OFFSET])
                ),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ShelfSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0, 0, cls.LOWER_OFFSET])
                ),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=ShelfSetting,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0, 0, cls.BOTTOM_OFFSET])
                ),
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
                child_type=BigBowlOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            # Everything that appears in other shelf layers.
            ProductionRule(
                child_type=ShelfSetting,
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
                child_type=CokeCanOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=TeaBottleOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            # Apples
            ProductionRule(
                child_type=AppleOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Apple1OrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Apple2OrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Apple3OrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            # Pears
            ProductionRule(
                child_type=PearOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Pear1OrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Pear2OrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            # Avocados
            ProductionRule(
                child_type=AvocadoOrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Avocado1OrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Avocado2OrNull,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
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
                p=0.6, max_children=10, start_at_one=False
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
                p=0.6, max_children=10, start_at_one=False
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


class Apple(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.033

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/apples/gala_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class AppleOrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Apple,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Apple.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Apple.SAVE_RADIUS,
                            0.04,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Apple.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Apple.SAVE_RADIUS,
                            0.04001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
        ]
        return rules


class Apple1(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.035

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/apples/golden_delicious_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Apple1OrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Apple1,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Apple1.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Apple1.SAVE_RADIUS,
                            0.04,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Apple1.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Apple1.SAVE_RADIUS,
                            0.04001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
        ]
        return rules


class Apple2(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.0375

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/apples/granny_smith_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Apple2OrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Apple2,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Apple2.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Apple2.SAVE_RADIUS,
                            0.04,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Apple2.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Apple2.SAVE_RADIUS,
                            0.04001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
        ]
        return rules


class Apple3(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.04

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/apples/fake_red_delicious_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Apple3OrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Apple3,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Apple3.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Apple3.SAVE_RADIUS,
                            0.045,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Apple3.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Apple3.SAVE_RADIUS,
                            0.04501,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
        ]
        return rules


class Pear(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.0375

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/pears/bose_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class PearOrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Pear,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Pear.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Pear.SAVE_RADIUS,
                            0.05,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Pear.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Pear.SAVE_RADIUS,
                            0.05001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class Pear1(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.034

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/pears/green_anjou_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Pear1OrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Pear1,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Pear1.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Pear1.SAVE_RADIUS,
                            0.05,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Pear1.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Pear1.SAVE_RADIUS,
                            0.05001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class Pear2(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.0375

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/pears/starkrimson_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Pear2OrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Pear2,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Pear2.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Pear2.SAVE_RADIUS,
                            0.04,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Pear2.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Pear2.SAVE_RADIUS,
                            0.04001,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
        ]
        return rules


class Avocado(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.0325

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/avocados/fuerte_avocado.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class AvocadoOrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Avocado,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Avocado.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Avocado.SAVE_RADIUS,
                            0.05,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Avocado.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Avocado.SAVE_RADIUS,
                            0.0501,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class Avocado1(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.03

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/avocados/hass_avocado.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Avocado1OrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Avocado1,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Avocado1.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Avocado1.SAVE_RADIUS,
                            0.05,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Avocado1.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Avocado1.SAVE_RADIUS,
                            0.0501,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class Avocado2(TerminalNode):
    KEEP_OUT_RADIUS = 0.02
    SAVE_RADIUS = 0.041

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/avocados/lula_avocado.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Avocado2OrNull(RepeatingSetNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=10, start_at_one=False
            ),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Avocado2,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor(
                        [
                            -Shelf.WIDTH / 2 + Avocado2.SAVE_RADIUS,
                            -Shelf.LENGTH / 2 + Avocado2.SAVE_RADIUS,
                            0.06,
                        ]
                    ),
                    ub=torch.tensor(
                        [
                            Shelf.WIDTH / 2 - Avocado2.SAVE_RADIUS,
                            Shelf.LENGTH / 2 - Avocado2.SAVE_RADIUS,
                            0.0601,
                        ]
                    ),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


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
