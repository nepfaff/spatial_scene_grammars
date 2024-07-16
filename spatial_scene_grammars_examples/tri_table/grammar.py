import torch
from pydrake.all import RigidTransform, RollPitchYaw

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *

""" 
Table (no geometry, static directive added after) -> plate_settings & (plate | null)*2 &
        (bowl | null)*2 & (cereal_box | null)*3 & (jug | null)*2
plate_settings -> plate_setting & plate_setting
plate_setting -> main_plate | main_bowl | null

main_plate -> plate & ontop_plate & left_of_plate & right_of_plate & top_of_plate & top_right_of_plate
ontop_plate (not visible) -> toast | apple | pear | bowl | null
left_of_plate (not visible) -> fork | null
right_of_plate (not visible) -> knive | null
top_of_plate (not visible) -> spoon | null
top_right_of_plate -> mug | null

main_bowl -> bowl & ontop_bowl & left_of_bowl & right_of_bowl & top_of_bowl
ontop_bowl (not visible) -> toast | apple | pear | null
right_of_bowl (not visible) -> spoon | null
top_of_bowl (not visible) -> spoon | null
<= Note: Can we use constraints to only ever produce <=1 spoon for each main_bowl?

plate -> plate | apple | pear | toast | null
bowl -> bowl | apple | pear | toast | null

dish_rack -> null
utensil_crock -> null
mug_holder -> null
cereal_box -> null
jug -> null
apple -> null
pear -> null
toast -> toast | null
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


class Table(AndNode):
    # Width and height in meters.
    WIDTH = 0.715
    HEIGHT = 1.4

    def __init__(self, tf):
        # NOTE: Static table geometry is added at simulation time using a welded
        # directive.
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlateSettings,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.zeros(3), variance=torch.tensor([1e-16, 0.2**2, 1e-16])
                ),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedPlates,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedBowls,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedCerealBoxes,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=SharedJugs,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]


class PlateSettings(AndNode):
    DISTANCE_FROM_CENTER = 0.2

    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=PlateSetting,
                xyz_rule=ParentFrameGaussianOffsetRule(  # TODO: Replace with uniform?
                    mean=torch.tensor([-cls.DISTANCE_FROM_CENTER, 0.0, 0.0]),
                    variance=torch.tensor([0.02**2, 0.02**2, 1e-16]),
                ),
                rotation_rule=SameRotationRule(
                    offset=torch.from_numpy(
                        RollPitchYaw(0, 0, np.pi).ToRotationMatrix().matrix()
                    )
                ),
            ),
            ProductionRule(
                child_type=PlateSetting,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([cls.DISTANCE_FROM_CENTER, 0.0, 0.0]),
                    variance=torch.tensor([0.02**2, 0.02**2, 1e-16]),
                ),
                rotation_rule=SameRotationRule(),
            ),
        ]


class PlateSetting(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.7, 0.2, 0.1]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=MainPlate,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=MainBowl,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class MainPlate(AndNode):
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Plate,
                xyz_rule=SamePositionRule(),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=OnTopMainPlate,
                xyz_rule=SamePositionRule(
                    offset=torch.tensor([0.0, 0.0, 0.018])
                ),  # Plate has height of 0.018m
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=LeftOfPlate,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, -0.1, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=RightOfPlate,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.1, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=TopOfPlate,
                xyz_rule=SamePositionRule(offset=torch.tensor([-0.1, 0.0, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=TopRightOfPlate,
                xyz_rule=SamePositionRule(offset=torch.tensor([-0.1, 0.1, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class OnTopMainPlate(OrNode):
    # Plate has a diameter of 0.2m.

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.1, 0.1, 0.2, 0.4]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Toast,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.008]),
                    variance=torch.tensor([0.03**2, 0.03**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Apple,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.03]),
                    variance=torch.tensor([0.025**2, 0.025**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Pear,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.035]),
                    variance=torch.tensor([0.025**2, 0.025**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Bowl,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, -0.009]),
                    variance=torch.tensor([0.02**2, 0.02**2, 1e-16]),
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


class LeftOfPlate(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.7, 0.3]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Fork,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.05, -0.007, 0.015]),
                    variance=torch.tensor([0.01**2, 1e-16, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(), np.array([1e6, 1e6, 100])
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class RightOfPlate(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.8, 0.2]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Knive,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.05, 0.007, 0.0075]),
                    variance=torch.tensor([0.01**2, 1e-16, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RollPitchYaw(np.array([np.pi, 0.0, 0.0])).ToRotationMatrix(),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class TopOfPlate(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.2, 0.6]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Spoon,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-0.015, 0.025, 0.013]),
                    variance=torch.tensor([1e-16, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RollPitchYaw(np.array([0.0, 0.0, np.pi / 2.0])).ToRotationMatrix(),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=TeaSpoon,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-0.015, 0.025, 0.013]),
                    variance=torch.tensor([1e-16, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RollPitchYaw(np.array([0.0, 0.0, np.pi / 2.0])).ToRotationMatrix(),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class TopRightOfPlate(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.8, 0.2]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Mug,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-0.03, 0.03, 0.0]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
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


class MainBowl(AndNode):
    # Bowl has a diameter of 0.145m and height of 0.065m.
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Bowl,
                xyz_rule=SamePositionRule(),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=OnTopBowl,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.0, 0.065])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=RightOfBowl,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.0725, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
            ProductionRule(
                child_type=TopOfBowl,
                xyz_rule=SamePositionRule(offset=torch.tensor([-0.0725, 0.0, 0.0])),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class OnTopBowl(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.1, 0.1, 0.1, 0.7]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Toast,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.008]),
                    variance=torch.tensor([0.02**2, 0.02**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Apple,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.0]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Pear,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.0]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
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


class RightOfBowl(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([1 / 6, 1 / 6, 2 / 3]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Spoon,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.05, 0.01, 0.01]),
                    variance=torch.tensor([0.01**2, 1e-16, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=TeaSpoon,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.05, 0.01, 0.01]),
                    variance=torch.tensor([0.01**2, 1e-16, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RotationMatrix(),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class TopOfBowl(OrNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([1 / 6, 1 / 6, 2 / 3]),
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Spoon,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-0.01, 0.02, 0.01]),
                    variance=torch.tensor([1e-16, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RollPitchYaw(np.array([0.0, 0.0, np.pi / 2.0])).ToRotationMatrix(),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=TeaSpoon,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([-0.01, 0.02, 0.01]),
                    variance=torch.tensor([1e-16, 0.01**2, 1e-16]),
                ),
                rotation_rule=ParentFrameBinghamRotationRule.from_rotation_and_rpy_variances(
                    RollPitchYaw(np.array([0.0, 0.0, np.pi / 2.0])).ToRotationMatrix(),
                    np.array([1e6, 1e6, 100]),
                ),  # Allow some yaw rotation.
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class SharedPlates(RepeatingSetNode):

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.7, max_children=2, start_at_one=False
            ),  # No plate with probability of ~0.7
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SharedPlate,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor([-0.3, -0.6, 0.0]),
                    ub=torch.tensor([0.3, 0.6, 0.001]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedBowls(RepeatingSetNode):

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.7, max_children=2, start_at_one=False
            ),  # No bowl with probability of ~0.7
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=SharedBowl,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor([-0.3, -0.6, 0.0]),
                    ub=torch.tensor([0.3, 0.6, 0.001]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedCerealBoxes(RepeatingSetNode):

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.3, max_children=3, start_at_one=False
            ),  # No cereal box with probability of ~0.3
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=CerealBox,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor([-0.3, -0.6, 0.001]),
                    ub=torch.tensor([0.3, 0.6, 0.002]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedJugs(RepeatingSetNode):

    def __init__(self, tf):
        super().__init__(
            tf=tf,
            rule_probs=RepeatingSetNode.get_geometric_rule_probs(
                p=0.6, max_children=2, start_at_one=False
            ),  # No jug with probability of ~0.6
            physics_geometry_info=None,
            observed=False,
        )

    @classmethod
    def generate_rules(cls):
        return [
            ProductionRule(
                child_type=Jug,
                xyz_rule=WorldFrameBBoxOffsetRule.from_bounds(
                    lb=torch.tensor([-0.3, -0.6, 0.091]),
                    ub=torch.tensor([0.3, 0.6, 0.092]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            )
        ]


class SharedPlate(OrNode):
    KEEP_OUT_RADIUS = 0.11

    def __init__(self, tf):
        # TODO: Load inertia from file and add to geom
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/plates/carlisle_plate_mesh_collision.sdf",
        )
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.5, 0.25, 0.25]),
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
                    variance=torch.tensor([0.02**2, 0.02**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Apple,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.043]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Pear,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.048]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class SharedBowl(OrNode):
    KEEP_OUT_RADIUS = 0.08

    def __init__(self, tf):
        # TODO: Load inertia from file and add to geom
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/bowls/ikea_dinera_bowl.sdf",
        )
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.4, 0.4]),
            physics_geometry_info=geom,
            observed=True,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Toast,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.073]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Apple,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.06]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Pear,
                xyz_rule=ParentFrameGaussianOffsetRule(
                    mean=torch.tensor([0.0, 0.0, 0.06]),
                    variance=torch.tensor([0.015**2, 0.015**2, 1e-16]),
                ),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
        ]
        return rules


class Toast(OrNode):
    # Height of 0.016m, length of 0.104m
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/food/sandwich/fake_toasted_bread_slice_mesh_collision.sdf",
        )
        super().__init__(
            tf=tf,
            rule_probs=torch.tensor([0.2, 0.8]),
            physics_geometry_info=geom,
            observed=True,
        )

    @classmethod
    def generate_rules(cls):
        rules = [
            ProductionRule(
                child_type=Toast,
                xyz_rule=SamePositionRule(offset=torch.tensor([0.0, 0.0, 0.016])),
                rotation_rule=ARBITRARY_YAW_ROTATION_RULE,
            ),
            ProductionRule(
                child_type=Null,
                xyz_rule=SamePositionRule(),
                rotation_rule=SameRotationRule(),
            ),
        ]
        return rules


class Apple(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/apples/gala_apple.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Pear(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/fruits/pears/bose_pear.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Mug(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/mug_2/mug_inomata_white_issue11152_mesh_collision.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Fork(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/forks/cambridge_jubilee_stainless_plastic_fork.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Knive(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/knives/cambridge_jubilee_stainless_plastic_knife.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Spoon(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/spoons/cambridge_jubilee_stainless_plastic_dinner_spoon.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class TeaSpoon(TerminalNode):
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/spoons/cambridge_jubilee_stainless_plastic_teaspoon.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Plate(TerminalNode):
    # Height of 0.018m.
    KEEPOUT_RADIUS = 0.1  # 20cm diameter

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/plates/ikea_dinera_plate_8in_mesh_collision.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Bowl(TerminalNode):
    # Diameter of 0.145m.
    # Height of 0.065m.
    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/home_kitchen/bowls/ikea_dinera_bowl.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class CerealBox(TerminalNode):
    KEEP_OUT_RADIUS = 0.15

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/food/cereal/punyos_cereal_box.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Jug(TerminalNode):
    KEEP_OUT_RADIUS = 0.06

    def __init__(self, tf):
        geom = PhysicsGeometryInfo(fixed=False)
        geom.register_model_file(
            drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
            "package://anzu/models/mug_2/mug_jeans_32oz_blue_issue5817.sdf",
        )
        super().__init__(tf=tf, physics_geometry_info=geom, observed=True)


class Null(TerminalNode):
    def __init__(self, tf):
        super().__init__(tf=tf, physics_geometry_info=None, observed=False)


class ObjectsOnTableConstraint(StructureConstraint):
    def __init__(self):
        lb = torch.tensor([-Table.WIDTH / 2.0 + 0.1, -Table.HEIGHT / 2.0 + 0.1, -0.02])
        ub = torch.tensor([Table.WIDTH / 2.0 - 0.1, Table.HEIGHT / 2.0 - 0.1, 1.0])
        super().__init__(lower_bound=lb, upper_bound=ub)

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
        xyzs = []  # in parent table frame
        for table in tables:
            # Collect table children xyz poses in table frame
            objs = scene_tree.get_children_recursive(table)
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


class MinNumObjectsConstraint(StructureConstraint):
    def __init__(self, min_num_objects):
        super().__init__(
            lower_bound=torch.tensor([min_num_objects]),
            upper_bound=torch.tensor([torch.inf]),
        )

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
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


class SharedObjectsNotInCollisionWithPlateSettingsConstraint(StructureConstraint):
    def __init__(self):
        super().__init__(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([torch.inf])
        )

    def eval(self, scene_tree):
        tables = scene_tree.find_nodes_by_type(Table)
        for table in tables:
            # Get shared objects.
            shared_plates = scene_tree.get_children_recursive(table, SharedPlates)
            shared_plate_objects = (
                np.array([scene_tree.get_children(p) for p in shared_plates])
                .flatten()
                .tolist()
            )
            shared_bowls = scene_tree.get_children_recursive(table, SharedBowls)
            shared_bowl_objects = (
                np.array([scene_tree.get_children(p) for p in shared_bowls])
                .flatten()
                .tolist()
            )
            shared_cereal_boxes = scene_tree.get_children_recursive(
                table, SharedCerealBoxes
            )
            shared_cereal_box_objects = (
                np.array([scene_tree.get_children(p) for p in shared_cereal_boxes])
                .flatten()
                .tolist()
            )
            shared_jugs = scene_tree.get_children_recursive(table, SharedJugs)
            shared_jug_objects = (
                np.array([scene_tree.get_children(p) for p in shared_jugs])
                .flatten()
                .tolist()
            )
            shared_objects = (
                shared_plate_objects
                + shared_bowl_objects
                + shared_cereal_box_objects
                + shared_jug_objects
            )

            # Get plate settings and associated objects.
            plate_settings = scene_tree.get_children_recursive(table, PlateSettings)
            plate_setting_objects = (
                np.array([scene_tree.get_children_recursive(p) for p in plate_settings])
                .flatten()
                .tolist()
            )
            # Remove unoberseved objects.
            plate_setting_objects = [
                obj for obj in plate_setting_objects if obj.observed
            ]

            if len(plate_setting_objects) == 0 or len(shared_objects) == 0:
                # Constraint trivially satisfied.
                return torch.tensor([1.0])

            # Extract translations and keep out radii.
            obj_translations = torch.stack(
                [obj.translation[:2] for obj in plate_setting_objects]
            )
            shared_translations = torch.stack(
                [shared_obj.translation[:2] for shared_obj in shared_objects]
            )
            keep_out_radii = torch.tensor(
                [shared_obj.KEEP_OUT_RADIUS for shared_obj in shared_objects]
            )

            # Compute the distance matrix between all pairs of objects and shared
            # objects.
            distances = torch.cdist(obj_translations, shared_translations)

            # Negative if any object is within keep out radius of any shared object.
            return torch.flatten(distances - keep_out_radii).unsqueeze(1)
