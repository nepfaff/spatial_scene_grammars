import unittest
import pytest

from pydrake.all import (BodyIndex, FrameIndex, ModelInstanceIndex)
import numpy as np

import torch
import pyro

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.serialization_model_directive import *
from spatial_scene_grammars.serialization_sdf import *

from spatial_scene_grammars_examples.kitchen.grammar_room_layout import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)

@pytest.fixture(params=range(10))
def trace(request):
    pyro.clear_param_store()
    torch.manual_seed(request.param)

    model = lambda: SceneTree.forward_sample_from_root_type(
        root_node_type=Kitchen,
        tf=torch.eye(4),
        name="kitchen"
    )
    return pyro.poutine.trace(model).get_trace()

@pytest.fixture()
def scene_tree(trace):
    return trace.nodes["_RETURN"]["value"]


def test_finite_logprob(trace):
    assert torch.isfinite(trace.log_prob_sum())

def test_simulation_setup(scene_tree):
    # Simulate the resulting scene.
    # May not expect this to work if the scene is generated in an
    # infeasible config... but short sim horizons seem not to error out,
    # so at least it checks the construction machinery and that Drake doesn't
    # hate the model on a fundamental level.
    simulate_scene_tree(scene_tree, 0.001, timestep=0.001, with_meshcat=False)

def test_model_directive_serialize_deserialize(scene_tree, subtests):
    # Can we serialize + deserialize to the model directive format,
    # and recover the same scene (as far as a Drake MBP is concerned)?
    if "ROS_PACKAGE_PATH" not in os.environ.keys():
        os.environ["ROS_PACKAGE_PATH"] = ""
    os.environ['ROS_PACKAGE_PATH'] = os.environ['ROS_PACKAGE_PATH'] + ":/tmp/pkg"
    with subtests.test("serialization"):
        # Make sure we can spit out the appropriate files.
        serialize_scene_tree_to_package_and_model_directive(
            scene_tree,
            package_name='test_save',
            package_parent_dir="/tmp/pkg/",
            remove_directory=True)
        assert os.path.isfile("/tmp/pkg/test_save/package.xml")
        assert os.path.isfile("/tmp/pkg/test_save/scene_tree.yaml")
        assert os.path.isdir("/tmp/pkg/test_save/sdf/")
        # This, at least, should always generate, since a floor always generates.
        assert os.path.isfile("/tmp/pkg/test_save/sdf/Floor_0::model_prims.sdf")
    with subtests.test("deserialization"):
        # Make sure we can load it back in and simulate it.
        mbp_wrangler = PackageToMbpAndSgBuilder(package_dir="/tmp/pkg/test_save")
        mbp_wrangler.Finalize()
    with subtests.test("mbp_matching"):
        # Double check the set of bodies and frames are identical, and
        # all frames are at the same position.
        _, orig_mbp, _ = compile_scene_tree_to_mbp_and_sg(scene_tree)
        orig_mbp.Finalize()
        loaded_mbp = mbp_wrangler.mbp

        orig_context = orig_mbp.CreateDefaultContext()
        loaded_context = loaded_mbp.CreateDefaultContext()
        
        print("Original has body names: ")
        for model_k in range(orig_mbp.num_model_instances()):
            model_id = ModelInstanceIndex(model_k)
            model_name = orig_mbp.GetModelInstanceName(model_id)
            print("\t", model_name)

        print("Loaded has model names: ")
        for model_k in range(loaded_mbp.num_model_instances()):
            model_id = ModelInstanceIndex(model_k)
            model_name = loaded_mbp.GetModelInstanceName(model_id)
            print("\t", model_name)

        print("Original has body names: ")
        for body_k in range(orig_mbp.num_bodies()):
            body_id = BodyIndex(body_k)
            body_name = orig_mbp.get_body(body_id).name()
            print("\t", body_name)

        print("Loaded has model names: ")
        for body_k in range(loaded_mbp.num_bodies()):
            body_id = BodyIndex(body_k)
            body_name = loaded_mbp.get_body(body_id).name()
            print("\t", body_name)

        # I've tried hard to make sure the serialized model re-loads
        # with the same order of models, so we can make sure each body
        # re-loads correctly.
        for model_k in range(orig_mbp.num_model_instances()):
            model_id = ModelInstanceIndex(model_k)
            model_name = orig_mbp.GetModelInstanceName(model_id)
            assert loaded_mbp.HasModelInstanceNamed(model_name), \
                            "Missing model %s" % model_name
            corresponding_model_id = loaded_mbp.GetModelInstanceByName(model_name)
            expected_body_ids = orig_mbp.GetBodyIndices(model_id)
            for body_id in expected_body_ids:
                body = orig_mbp.get_body(body_id)
                assert loaded_mbp.HasBodyNamed(name=body.name(), model_instance=corresponding_model_id), \
                                "Missing body %s in model %s" % (body.name(), model_name)
                corresponding_body = loaded_mbp.GetBodyByName(body.name(), model_instance=corresponding_model_id)
                # Make sure they're at the same world pose
                body_tf = orig_mbp.EvalBodyPoseInWorld(orig_context, body)
                corresponding_body_tf = loaded_mbp.EvalBodyPoseInWorld(loaded_context, corresponding_body)
                error_tf = body_tf.multiply(corresponding_body_tf.inverse())
                if not np.allclose(error_tf.GetAsMatrix4(), np.eye(4), 1E-6):
                    print("Body pose mismatch for body %s in model %s: \n\t%s\n\t\tvs\n\t%s\n\t\tErr:\n\t%s" % (
                          body.name(), model_name,
                          str(body_tf.GetAsMatrix4()),
                          str(corresponding_body_tf.GetAsMatrix4()),
                          str(error_tf.GetAsMatrix4())))
                    assert False, "See test output -- body pose mismatch during deserialize."



    with subtests.test("simulation"):
        # Set up and run some brief sim, and make sure Drake is OK with it, and visualize for human
        # check.

        visualizer = ConnectMeshcatVisualizer(mbp_wrangler.builder, mbp_wrangler.scene_graph,
            zmq_url="new", open_browser=False)
        diagram = mbp_wrangler.builder.Build()
        diag_context = diagram.CreateDefaultContext()
        sim = Simulator(diagram)
        sim.set_target_realtime_rate(1.0)
        sim.AdvanceTo(0.001)

def test_sdf_serialize_deserialize(scene_tree, subtests):
    # Can we serialize + deserialize to the model directive format,
    # and recover the same scene (as far as a Drake MBP is concerned)?
    if "ROS_PACKAGE_PATH" not in os.environ.keys():
        os.environ["ROS_PACKAGE_PATH"] = ""
    os.environ['ROS_PACKAGE_PATH'] = os.environ['ROS_PACKAGE_PATH'] + ":/tmp/pkg"
    out_sdf_name = "/tmp/out_sdf.sdf"
    with subtests.test("serialization"):
        # Make sure we can spit out the appropriate files.
        serialize_scene_tree_to_package_and_single_sdf(
            scene_tree,
            out_sdf_name=out_sdf_name,
            include_static_tag=False, 
            include_model_files=True,
            pybullet_compat=False
        )
        assert os.path.isfile(out_sdf_name)

    with subtests.test("deserialization"):
        # Make sure we can load it back in and simulate it.
        builder = DiagramBuilder()
        loaded_mbp, loaded_scene_graph = AddMultibodyPlantSceneGraph(builder, 0.001)
        parser = Parser(loaded_mbp)
        loaded_model_ids = parser.AddAllModelsFromFile(out_sdf_name)

    with subtests.test("simulation"):
        # Set up and run some brief sim, and make sure Drake is OK with it, and visualize for human
        # check.

        visualizer = ConnectMeshcatVisualizer(builder, loaded_scene_graph,
            zmq_url="new", open_browser=False)
        loaded_mbp.Finalize()
        diagram = builder.Build()
        diag_context = diagram.CreateDefaultContext()
        sim = Simulator(diagram)
        sim.set_target_realtime_rate(1.0)
        sim.AdvanceTo(0.001)


if __name__ == '__main__':
    pytest.main()