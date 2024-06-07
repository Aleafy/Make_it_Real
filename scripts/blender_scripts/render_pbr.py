"""
Script to run within Blender to render a 3D model as RGBAD images.

Example usage

    blender paste_pbr.blend -b -P render_pbr.py -- \
        --input_path ../../data/mesh/gold_gun/gold_gun.obj \
        --texture_path ../../experiments/gold_gun_train/mat_whole_pbr    \
        --output_path ../../output/refine_output
"""

import argparse
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector
from mathutils.noise import random_unit_vector
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
from PIL import Image
import numpy as np

MAX_DEPTH = 5.0
FORMAT_VERSION = 6
UNIFORM_LIGHT_DIRECTION = [0.09387503, -0.63953443, -0.7630093]

def CleanScene():
    print("cleaning scene")
    # Clear existing lights in the scene
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.cameras,
            bpy.data.textures,
            bpy.data.images
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)
    print("=================Cleaning scene deleting all objects==================================")     
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    while len(bpy.data.images) > 0:
        bpy.data.images.remove(bpy.data.images[0])

    allMeshes=[]
    for mes in bpy.data.meshes:
       allMeshes.append(mes)
       print("Deleting Mesh:")
       print(mes)
    for mes in allMeshes:
        bpy.data.meshes.remove(mes)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    print("Done cleaning")

def AddBackground(hdr_pth):
    bpy.context.scene.world= bpy.data.worlds['World']

    world = bpy.data.worlds['World']

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in nodes:
        nodes.remove(node)

    bg_node = nodes.new(type='ShaderNodeTexEnvironment')
    bg_node.name = 'Background.001'
    bg_node.location = (0, 0) 

    bg_node.image= bpy.data.images.load(filepath=hdr_pth)

    world_output_node = nodes.new(type='ShaderNodeOutputWorld')
    world_output_node.name = 'World' 
    world_output_node.location = (5000, 0) 

    links.new(bg_node.outputs['Color'], world_output_node.inputs['Surface'])

def paste_mat(args):
    obj_path = args.input_path

    CleanScene()

    bpy.ops.import_scene.obj(filepath=obj_path, use_edges=True, use_smooth_groups=True, use_split_objects=True, use_split_groups=True, use_groups_as_vgroups=False, use_image_search=True, split_mode='ON', global_clamp_size=0)

    Obj = bpy.context.selected_objects[0]

    mat = Obj.material_slots[0].material

    base_color_path = None
    normal_map_path = None
    
    if Obj.material_slots and Obj.material_slots[0].material:
        mat = Obj.material_slots[0].material
        
        if mat.node_tree:
            nodes = mat.node_tree.nodes
            base_color_img = None
            normal_map_img = None

            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if node.inputs['Base Color'].links:
                        base_color_node = node.inputs['Base Color'].links[0].from_node
                        if base_color_node.type == 'TEX_IMAGE':
                            base_color_img = base_color_node.image
                            if base_color_img:
                                base_color_path = base_color_img.filepath
                                print("Base Color Path:", base_color_path)

                    if node.inputs['Normal'].links:
                        normal_map_node = node.inputs['Normal'].links[0].from_node
                        if normal_map_node.type == 'NORMAL_MAP':
                            if normal_map_node.inputs['Color'].links:
                                image_node = normal_map_node.inputs['Color'].links[0].from_node
                                if image_node.type == 'TEX_IMAGE':
                                    normal_map_img = image_node.image
                                    if normal_map_img:
                                        normal_map_path = normal_map_img.filepath
                                        print("Normal Map Path:", normal_map_path)

            if base_color_img and normal_map_img:
                print("search success")
            else:
                print("some images do not exist")
        else:
            print("there is no bsdf tree")
    else:
        print("object do not have any material")

    bpy.context.view_layer.objects.active = Obj
    print(len(Obj.material_slots))
    for mat_slot in Obj.material_slots:
        bpy.context.object.active_material_index = 0
        bpy.ops.object.material_slot_remove()

        
    # get material (bind to fake user)
    material_name = "MaterialTemplate"
    material = bpy.data.materials.get(material_name)


    if material and Obj:
        if not Obj.data.materials:
            Obj.data.materials.append(material)
        else:
            Obj.data.materials[0]=material

    # change material
    # textures_dir = "experiments/obj_train/mat_whole_refine"
    textures_dir = args.texture_path

    nodes = material.node_tree.nodes
    if base_color_path is not None:
        if base_color_path[0] != '.':
            base_color_path = '.' + base_color_path
        nodes['Base Color'].image = bpy.data.images.load(base_color_path)
    else:
        nodes['Base Color'].image = bpy.data.images.load(os.path.join(textures_dir, 'basecolor.png'))
    if normal_map_path is not None:
        if normal_map_path[0] != '.':
            normal_map_path = '.' + normal_map_path
        nodes['Normal'].image = bpy.data.images.load(normal_map_path)
    else:
        nodes['Normal'].image = bpy.data.images.load(os.path.join(textures_dir, 'normal.png'))

    for Fname in os.listdir(textures_dir):
        if ("oughness." in Fname) or ("ROUGH." in Fname) or ("roughness" in Fname) or ("ROUGHNESS" in Fname) or ("roughnness" in Fname):
            nodes['Roughness'].image = bpy.data.images.load(os.path.join(textures_dir, Fname))
        # if ("ormal." in Fname)  or ("NORM." in Fname) or ("normal" in Fname)  or ("NORMAL" in Fname) or ("Normal" in Fname):
        #     nodes['Normal'].image = bpy.data.images.load(os.path.join(textures_dir, Fname))
        if ("DISP." in Fname) or ("displacement" in Fname):
            nodes['Displacement'].image = bpy.data.images.load(os.path.join(textures_dir, Fname))
        if ("etallic." in Fname) or ("etalness." in Fname)  or ("etal." in Fname) or ("etalic." in Fname) :
            nodes['Metallic'].image = bpy.data.images.load(os.path.join(textures_dir, Fname))
        if ("pecular."  in Fname):
            nodes['Specular'].image = bpy.data.images.load(os.path.join(textures_dir, Fname))

def clear_lights():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Light):
            obj.select_set(True)
    bpy.ops.object.delete()

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def rotate_to_zzz():
    for obj in scene_meshes():
        obj.rotation_euler = [ math.radians(0), math.radians(0), math.radians(90)]
        # obj.rotation_euler = [ math.radians(90), math.radians(260), math.radians(90)]

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene(rotate=False):
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)

    

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
        # obj.rotation_euler[0] = 0
        
    if rotate:
        rotate_to_zzz()

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def create_camera():
    # https://b3d.interplanety.org/en/how-to-create-camera-through-the-blender-python-api/
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object


def set_camera(direction, camera_dist=2.0):
    camera_pos = -camera_dist * direction
    bpy.context.scene.camera.location = camera_pos

    # https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    rot_quat = direction.to_track_quat("-Z", "Y")
    bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()

    bpy.context.view_layer.update()


def randomize_camera(camera_dist=2.0):
    direction = random_unit_vector()
    set_camera(direction, camera_dist=camera_dist)


def pan_camera(time, axis="Z", camera_dist=2.0, elevation=-0.1):
    angle = time * math.pi * 2 + 0.25 * math.pi
    direction = [-math.cos(angle), -math.sin(angle), -elevation]
    assert axis in ["X", "Y", "Z"]
    if axis == "X":
        direction = [direction[2], *direction[:2]]
    elif axis == "Y":
        direction = [direction[0], -elevation, direction[1]]
    direction = Vector(direction).normalized()
    set_camera(direction, camera_dist=camera_dist)


def place_camera(time, camera_pose_mode="random", camera_dist_min=2.0, camera_dist_max=2.0):
    camera_dist = random.uniform(camera_dist_min, camera_dist_max)
    if camera_pose_mode == "random":
        randomize_camera(camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular":
        pan_camera(time, axis="Z", camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular-elevated":
        pan_camera(time, axis="Z", camera_dist=camera_dist, elevation=0.2617993878)
    else:
        raise ValueError(f"Unknown camera pose mode: {camera_pose_mode}")


def create_light(location, energy=1.0, angle=0.5 * math.pi / 180):
    # https://blender.stackexchange.com/questions/215624/how-to-create-a-light-with-the-python-api-in-blender-2-92
    light_data = bpy.data.lights.new(name="Light", type="SUN")
    light_data.energy = energy
    light_data.angle = angle
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    direction = -location
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()

    bpy.context.collection.objects.link(light_object)
    light_object.location = location

def create_light_cap3d(name, light_type, energy, location, rotation):
    bpy.ops.object.light_add(type=light_type, align='WORLD', location=location, scale=(1, 1, 1))
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.rotation_euler = rotation
    return light

def three_point_lighting():
    
    # Key ligh
    key_light = create_light_cap3d(
        name="KeyLight",
        light_type='POINT',
        energy=2000,
        location=(4, -4, 4),
        rotation=(math.radians(45), 0, math.radians(45))
    )
    
    key_light2 = create_light_cap3d(
        name="KeyLight",
        light_type='POINT',
        energy=2000,
        location=(-4, -4, 2),
        rotation=(math.radians(45), 0, math.radians(45))
    )
    
    key_light3 = create_light_cap3d(
        name="KeyLight",
        light_type='POINT',
        energy=2000,
        location=(-4, -4, -2),
        rotation=(math.radians(45), 0, math.radians(45))
    )

    
def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object

# lighting_setting from cap3d
def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def create_random_lights(count=4, distance=2.0, energy=1.5):
    clear_lights()
    for _ in range(count):
        create_light(random_unit_vector() * distance, energy=energy)


def create_camera_light():
    clear_lights()
    create_light(bpy.context.scene.camera.location, energy=5.0)


def create_uniform_light(backend):
    clear_lights()
    # Random direction to decorrelate axis-aligned sides.
    pos = Vector(UNIFORM_LIGHT_DIRECTION)
    angle = 0.0092 if backend == "CYCLES" else math.pi
    create_light(pos, energy=5.0, angle=angle)
    create_light(-pos, energy=5.0, angle=angle)


def create_vertex_color_shaders():
    # By default, Blender will ignore vertex colors in both the
    # Eevee and Cycles backends, since these colors aren't
    # associated with a material.
    #
    # What we do here is create a simple material shader and link
    # the vertex color to the material color.
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, (bpy.types.Mesh)):
            continue

        if len(obj.data.materials):
            # We don't want to override any existing materials.
            continue

        color_keys = (obj.data.vertex_colors or {}).keys()
        if not len(color_keys):
            # Many objects will have no materials *or* vertex colors.
            continue

        mat = bpy.data.materials.new(name="VertexColored")
        mat.use_nodes = True

        # There should be a Principled BSDF by default.
        bsdf_node = None
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf_node = node
        assert bsdf_node is not None, "material has no Principled BSDF node to modify"

        socket_map = {}
        for input in bsdf_node.inputs:
            socket_map[input.name] = input

        # Make sure nothing lights the object except for the diffuse color.
        socket_map["Specular"].default_value = 0.0
        socket_map["Roughness"].default_value = 1.0

        v_color = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        v_color.layer_name = color_keys[0]

        mat.node_tree.links.new(v_color.outputs[0], socket_map["Base Color"])

        obj.data.materials.append(mat)


def create_default_materials():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            if not len(obj.data.materials):
                mat = bpy.data.materials.new(name="DefaultMaterial")
                mat.use_nodes = True
                obj.data.materials.append(mat)

def AddWhiteBackground():
    bpy.context.scene.world= bpy.data.worlds['World']

    world = bpy.data.worlds['World']

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in nodes:
        nodes.remove(node)

    bg_node = nodes.new(type='ShaderNodeBackground')
    bg_node.name = 'Background.001' 
    bg_node.location = (0, 0) 

    bg_node.inputs['Color'].default_value = (1, 1, 1, 1) 
    bg_node.inputs['Strength'].default_value = 0.5 

    world_output_node = nodes.new(type='ShaderNodeOutputWorld')
    world_output_node.name = 'World' 
    world_output_node.location = (5000, 0)

    links.new(bg_node.outputs['Background'], world_output_node.inputs['Surface'])

def find_materials():
    all_materials = set()
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, (bpy.types.Mesh)):
            continue
        for mat in obj.data.materials:
            all_materials.add(mat)
    return all_materials


def get_socket_value(tree, socket):
    default = socket.default_value
    if not isinstance(default, float):
        default = list(default)
    for link in tree.links:
        if link.to_socket == socket:
            return (link.from_socket, default)
    return (None, default)


def clear_socket_input(tree, socket):
    for link in list(tree.links):
        if link.to_socket == socket:
            tree.links.remove(link)


def set_socket_value(tree, socket, socket_and_default):
    clear_socket_input(tree, socket)
    old_source_socket, default = socket_and_default
    if isinstance(default, float) and not isinstance(socket.default_value, float):
        # Codepath for setting Emission to a previous alpha value.
        socket.default_value = [default] * 3 + [1.0]
    else:
        socket.default_value = default
    if old_source_socket is not None:
        tree.links.new(old_source_socket, socket)


def setup_nodes(output_path, capturing_material_alpha: bool = False):
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # Helpers to perform math on links and constants.
    def node_op(op: str, *args, clamp=False):
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op
        if clamp:
            node.use_clamp = True
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])
        return node.outputs[0]

    def node_clamp(x, maximum=1.0):
        return node_op("MINIMUM", x, maximum)

    def node_mul(x, y, **kwargs):
        return node_op("MULTIPLY", x, y, **kwargs)

    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene

    input_sockets = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]

        # We apply sRGB here so that our fixed-point depth map and material
        # alpha values are not sRGB, and so that we perform ambient+diffuse
        # lighting in linear RGB space.
        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        color_node.from_color_space = "Linear"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]
    split_node = tree.nodes.new(type="CompositorNodeSepRGBA")
    tree.links.new(color_socket, split_node.inputs[0])
    # Create separate file output nodes for every channel we care about.
    # The process calling this script must decide how to recombine these
    # channels, possibly into a single image.
    for i, channel in enumerate("rgba") if not capturing_material_alpha else [(0, "MatAlpha")]:
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = f"{output_path}_{channel}"
        links.new(split_node.outputs[i], output_node.inputs[0])

    if capturing_material_alpha:
        # No need to re-write depth here.
        return

    depth_out = node_clamp(node_mul(input_sockets["Depth"], 1 / MAX_DEPTH))
    output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    output_node.base_path = f"{output_path}_depth"
    links.new(depth_out, output_node.inputs[0])


def render_scene(output_path, fast_mode: bool):
    use_workbench = bpy.context.scene.render.engine == "BLENDER_WORKBENCH"
    if use_workbench:
        # We must use a different engine to compute depth maps.
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
        bpy.context.scene.eevee.taa_render_samples = 1  # faster, since we discard image.
    if fast_mode:
        if bpy.context.scene.render.engine == "BLENDER_EEVEE":
            bpy.context.scene.eevee.taa_render_samples = 1
        elif bpy.context.scene.render.engine == "CYCLES":
            bpy.context.scene.cycles.samples = 256
    else:
        if bpy.context.scene.render.engine == "CYCLES":
            # We should still impose a per-frame time limit
            # so that we don't timeout completely.
            bpy.context.scene.cycles.time_limit = 40
    bpy.context.view_layer.update()
    # bpy.context.scene.use_nodes = True
    # bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    # bpy.context.scene.view_settings.view_transform = "Raw"  # sRGB done in graph nodes
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    # bpy.context.scene.render.image_settings.file_format = "PNG"
    # bpy.context.scene.render.image_settings.color_mode = "BW"
    # bpy.context.scene.render.image_settings.color_depth = "16"
    bpy.context.scene.render.filepath = output_path
    # setup_nodes(output_path)
    bpy.ops.render.render(write_still=True)
    # name, ext = os.path.splitext(output_path)
    out_image = Image.open(output_path)
  

    if use_workbench:
        # Re-render RGBA using workbench with texture mode, since this seems
        # to show the most reasonable colors when lighting is broken.
        bpy.context.scene.use_nodes = False
        bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.image_settings.color_depth = "8"
        bpy.context.scene.display.shading.color_type = "TEXTURE"
        bpy.context.scene.display.shading.light = "FLAT"
        if fast_mode:
            # Single pass anti-aliasing.
            bpy.context.scene.display.render_aa = "FXAA"
        os.remove(output_path)
        bpy.ops.render.render(write_still=True)
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.color_depth = "16"
    
    return out_image


def scene_fov():
    x_fov = bpy.context.scene.camera.data.angle_x
    y_fov = bpy.context.scene.camera.data.angle_y
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    if bpy.context.scene.camera.data.angle == x_fov:
        y_fov = 2 * math.atan(math.tan(x_fov / 2) * height / width)
    else:
        x_fov = 2 * math.atan(math.tan(y_fov / 2) * width / height)
    return x_fov, y_fov


def write_camera_metadata(path):
    x_fov, y_fov = scene_fov()
    bbox_min, bbox_max = scene_bbox()
    matrix = bpy.context.scene.camera.matrix_world
    with open(path, "w") as f:
        json.dump(
            dict(
                format_version=FORMAT_VERSION,
                max_depth=MAX_DEPTH,
                bbox=[list(bbox_min), list(bbox_max)],
                origin=list(matrix.col[3])[:3],
                x_fov=x_fov,
                y_fov=y_fov,
                x=list(matrix.col[0])[:3],
                y=list(-matrix.col[1])[:3],
                z=list(-matrix.col[2])[:3],
            ),
            f,
        )


def save_rendering_dataset(
    input_path: str,
    output_path: str,
    num_images: int,
    backend: str,
    light_mode: str,
    camera_pose: str,
    camera_dist_min: float,
    camera_dist_max: float,
    fast_mode: bool,
    rotate: bool,
    hdr_pth: str
):
    assert light_mode in ["random", "uniform", "camera", "cap3d"]
    assert camera_pose in ["random", "z-circular", "z-circular-elevated"]
    # import_model(input_path)
    bpy.context.scene.render.engine = backend
    normalize_scene(rotate=rotate)
    os.makedirs(os.getcwd()+"/"+output_path, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=os.getcwd()+"/"+output_path+"/model.glb", export_format='GLB') 
    bpy.context.scene.render.film_transparent = True
    if light_mode == "random":
        create_random_lights()
    elif light_mode == "uniform":
        create_uniform_light(backend)
    elif light_mode == "cap_3d":
        pass
    # AddWhiteBackground
    # AddWhiteBackground()    
    #AddWhiteBackground()
    AddBackground(hdr_pth)
    three_point_lighting()
    create_camera()
    create_vertex_color_shaders()
    
    all_four_images = []
    for i in range(num_images):
        t = i / max(num_images, 1)  # same as np.linspace(0, 1, num_images)
        place_camera(
            t,
            camera_pose_mode=camera_pose,
            camera_dist_min=camera_dist_min,
            camera_dist_max=camera_dist_max,
        )
        if light_mode == "camera":
            create_camera_light()
        image = render_scene(
            os.path.join(output_path, f"{i:05}.png"),
            fast_mode=fast_mode,
        )
        all_four_images.append(image)

    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    def gen4():
        view_u = all_four_images[0]
        # for i in [0, 1, 4, 5]:
        view_u = get_concat_h(view_u, all_four_images[1])
        view_d = all_four_images[2]
        view_d = get_concat_h(view_d, all_four_images[3])
        view = get_concat_v(view_u, view_d)
        return view

    gen4().save(os.path.join(output_path, f"four_v.png"))
    #     write_camera_metadata(os.path.join(output_path, f"{i:05}.json"))
    # with open(os.path.join(output_path, "info.json"), "w") as f:
    #     info = dict(
    #         backend=backend,
    #         light_mode=light_mode,
    #         fast_mode=fast_mode,
    #         format_version=FORMAT_VERSION,
    #         channels=["R", "G", "B", "A", "D"],
    #         scale=0.5,  # The scene is bounded by [-scale, scale].
    #     )
    #     json.dump(info, f)


def main():
    try:
        dash_index = sys.argv.index("--")
    except ValueError as exc:
        raise ValueError("arguments must be preceded by '--'") from exc

    raw_args = sys.argv[dash_index + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--texture_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--backend", type=str, default="CYCLES")
    parser.add_argument("--light_mode", type=str, default="cap3d")
    parser.add_argument("--camera_pose", type=str, default="z-circular-elevated")
    parser.add_argument("--camera_dist_min", type=float, default=2.0)
    parser.add_argument("--camera_dist_max", type=float, default=2.0)
    parser.add_argument("--hdr_pth", type=str, default='data/hdr/poly_haven_studio_2k.hdr')
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args(raw_args)
    
    paste_mat(args)

    save_rendering_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        num_images=args.num_images,
        backend=args.backend,
        light_mode=args.light_mode,
        camera_pose=args.camera_pose,
        camera_dist_min=args.camera_dist_min,
        camera_dist_max=args.camera_dist_max,
        fast_mode=args.fast_mode,
        rotate=args.rotate,
        hdr_pth=args.hdr_pth
    )


main()