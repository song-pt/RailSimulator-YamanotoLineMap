"""Assemble an exported scene directory in Blender and render a cab-view frame range.

Run:  blender -b --python blender_scene.py -- SCENE_DIR TEX_DIR OUT_DIR [--frames A B] [--speed KMH]
          [--res W H] [--samples N] [--still S_METRES] [--save BLEND]
SCENE_DIR comes from export_scene.py (one .npy triple per material + scene.json).
"""
import json, math, os, sys
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
SCENE, TEX, OUT = argv[:3]
opt = {}
i = 3
while i < len(argv):
    k = argv[i].lstrip("-")
    if k in ("frames", "res"):
        opt[k] = (int(argv[i + 1]), int(argv[i + 2])); i += 3
    else:
        opt[k] = argv[i + 1]; i += 2
FPS = 24
SPEED = float(opt.get("speed", 70)) / 3.6

meta = json.load(open(os.path.join(SCENE, "scene.json")))

# ---------------------------------------------------------------- materials
def img(name, kind, non_color=False):
    path = os.path.join(TEX, "%s_%s.jpg" % (name, kind))
    if not os.path.exists(path):
        return None
    im = bpy.data.images.load(path, check_existing=True)
    if non_color:
        im.colorspace_settings.name = "Non-Color"
    return im


def pbr(mat, tex, size, tint=(1, 1, 1), rough=None, metal=0.0, normal=0.6):
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    uv = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / size, 1 / size, 1)
    nt.links.new(uv.outputs["UV"], mp.inputs["Vector"])
    d = img(tex, "diff")
    if d:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = d
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs[7].default_value = (*tint, 1)
        nt.links.new(t.outputs["Color"], mix.inputs[6])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        color_out = mix.outputs[2]
    else:
        bsdf.inputs["Base Color"].default_value = (*tint, 1)
        color_out = None
    r = img(tex, "rough", True)
    if r and rough is None:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = r
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.7 if rough is None else rough
    n = img(tex, "nor", True)
    if n and normal > 0:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = n
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = normal
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = metal
    return mp, color_out


def flat(mat, color, rough=0.6, metal=0.0, emit=None):
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*emit[0], 1)
        bsdf.inputs["Emission Strength"].default_value = emit[1]


def windows(mat, wall_tex, size, tint, bay=3.2, floor=3.6, glass_frac=(0.62, 0.55), shop=True):
    """Facade: textured wall with a procedural window grid driven by UV metres (u along wall, v height)."""
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    mp, col = pbr(mat, wall_tex, size, tint)
    uv = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs["Vector"])

    def math_node(op, a, b=None, val=None):
        m = nt.nodes.new("ShaderNodeMath"); m.operation = op
        if isinstance(a, (int, float)):
            m.inputs[0].default_value = a
        else:
            nt.links.new(a, m.inputs[0])
        if b is not None:
            if isinstance(b, (int, float)):
                m.inputs[1].default_value = b
            else:
                nt.links.new(b, m.inputs[1])
        return m.outputs[0]

    fu = math_node("FRACT", math_node("DIVIDE", sep.outputs["X"], bay))
    fv = math_node("FRACT", math_node("DIVIDE", sep.outputs["Y"], floor))
    gu = math_node("LESS_THAN", math_node("ABSOLUTE", math_node("SUBTRACT", fu, 0.5)), glass_frac[0] / 2)
    gv = math_node("LESS_THAN", math_node("ABSOLUTE", math_node("SUBTRACT", fv, 0.55)), glass_frac[1] / 2)
    win = math_node("MULTIPLY", gu, gv)
    above = math_node("GREATER_THAN", sep.outputs["Y"], 4.2)
    win = math_node("MULTIPLY", win, above)
    if shop:  # ground floor shop fronts
        low = math_node("LESS_THAN", sep.outputs["Y"], 3.6)
        sg = math_node("LESS_THAN", math_node("ABSOLUTE", math_node("SUBTRACT", fu, 0.5)), 0.42)
        win = math_node("MAXIMUM", win, math_node("MULTIPLY", low, math_node("MULTIPLY", sg, math_node("GREATER_THAN", sep.outputs["Y"], 0.3))))
    # per-window variation (some lit / curtains)
    cell = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(math_node("FLOOR", math_node("DIVIDE", sep.outputs["X"], bay)), cell.inputs["X"])
    nt.links.new(math_node("FLOOR", math_node("DIVIDE", sep.outputs["Y"], floor)), cell.inputs["Y"])
    wn = nt.nodes.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "3D"
    nt.links.new(cell.outputs["Vector"], wn.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.045, 0.055, 1)
    ramp.color_ramp.elements[1].color = (0.16, 0.18, 0.2, 1)
    nt.links.new(wn.outputs["Value"], ramp.inputs["Fac"])
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
    nt.links.new(win, mixc.inputs["Factor"])
    if col is not None:
        nt.links.new(col, mixc.inputs[6])
    else:
        mixc.inputs[6].default_value = (*tint, 1)
    nt.links.new(ramp.outputs["Color"], mixc.inputs[7])
    nt.links.new(mixc.outputs[2], bsdf.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMix"); mr.data_type = "FLOAT"
    nt.links.new(win, mr.inputs["Factor"]); mr.inputs[2].default_value = 0.8; mr.inputs[3].default_value = 0.06
    nt.links.new(mr.outputs[0], bsdf.inputs["Roughness"])
    mm = nt.nodes.new("ShaderNodeMix"); mm.data_type = "FLOAT"
    nt.links.new(win, mm.inputs["Factor"]); mm.inputs[2].default_value = 0.0; mm.inputs[3].default_value = 0.7
    nt.links.new(mm.outputs[0], bsdf.inputs["Metallic"])


def make_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if name.startswith("sign_"):
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        path = os.path.join(SCENE, "%s.png" % name)
        if os.path.exists(path):
            t = mat.node_tree.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(path)
            mat.node_tree.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
            mat.node_tree.links.new(t.outputs["Color"], bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = 0.6
        return mat
    P = {
        "rail": lambda m: pbr(m, "rust", 0.6, (0.55, 0.42, 0.35), rough=0.75, metal=0.6),
        "rail_head": lambda m: flat(m, (0.75, 0.75, 0.76), 0.22, 1.0),
        "ballast": lambda m: pbr(m, "ballast", 1.6, (0.62, 0.6, 0.58), normal=1.0),
        "slab_track": lambda m: pbr(m, "concrete_smooth", 3.0, (0.75, 0.75, 0.73)),
        "sleeper": lambda m: pbr(m, "sleeper", 1.5, (0.82, 0.8, 0.77)),
        "fastening": lambda m: flat(m, (0.12, 0.12, 0.12), 0.5, 0.6),
        "deck_top": lambda m: pbr(m, "concrete", 4.0, (0.55, 0.54, 0.52)),
        "brick": lambda m: pbr(m, "brick", 2.5, (1.0, 0.85, 0.8)),
        "concrete": lambda m: pbr(m, "concrete", 4.0, (0.85, 0.84, 0.82)),
        "girder": lambda m: pbr(m, "metal", 2.0, (0.35, 0.4, 0.45), metal=0.5),
        "parapet": lambda m: pbr(m, "concrete_smooth", 2.0, (0.8, 0.8, 0.78)),
        "platform": lambda m: pbr(m, "platform", 2.0, (0.75, 0.75, 0.74)),
        "platform_side": lambda m: pbr(m, "concrete", 3.0, (0.7, 0.7, 0.68)),
        "edge_white": lambda m: flat(m, (0.85, 0.85, 0.82), 0.5),
        "tactile": lambda m: flat(m, (0.9, 0.72, 0.05), 0.55),
        "psd_frame": lambda m: flat(m, (0.78, 0.8, 0.82), 0.35, 0.3),
        "column": lambda m: flat(m, (0.7, 0.72, 0.74), 0.4, 0.4),
        "canopy_top": lambda m: pbr(m, "roof_sheet", 2.0, (0.6, 0.62, 0.64), metal=0.5),
        "canopy_under": lambda m: flat(m, (0.82, 0.83, 0.84), 0.6, 0.0, emit=((0.9, 0.92, 0.95), 0.25)),
        "canopy_fascia": lambda m: flat(m, (0.75, 0.77, 0.78), 0.4, 0.3),
        "metal": lambda m: flat(m, (0.5, 0.52, 0.55), 0.4, 0.7),
        "bench": lambda m: flat(m, (0.15, 0.45, 0.25), 0.45),
        "stair_pit": lambda m: flat(m, (0.08, 0.08, 0.09), 0.9),
        "mast": lambda m: flat(m, (0.55, 0.57, 0.58), 0.45, 0.6),
        "insulator": lambda m: flat(m, (0.45, 0.3, 0.2), 0.3),
        "wire_copper": lambda m: flat(m, (0.45, 0.32, 0.22), 0.35, 1.0),
        "wire_steel": lambda m: flat(m, (0.25, 0.26, 0.27), 0.4, 1.0),
        "asphalt": lambda m: pbr(m, "asphalt", 5.0, (0.6, 0.6, 0.6)),
        "sidewalk": lambda m: pbr(m, "pavement", 3.0, (0.85, 0.84, 0.82)),
        "road_mark": lambda m: flat(m, (0.85, 0.85, 0.85), 0.6),
        "water": lambda m: flat(m, (0.03, 0.06, 0.07), 0.05),
        "ground": lambda m: pbr(m, "pavement", 4.0, (0.7, 0.7, 0.68)),
        "roof": lambda m: pbr(m, "roof", 5.0, (0.7, 0.7, 0.7)),
        "facade_station_brick": lambda m: windows(m, "brick", 2.5, (1.0, 0.9, 0.85), bay=2.8, floor=4.5, shop=False),
        "facade_glass": lambda m: windows(m, "metal", 3.0, (0.55, 0.6, 0.65), bay=1.6, floor=4.0, glass_frac=(0.85, 0.75), shop=False),
        "facade_office_tile": lambda m: windows(m, "facade_tile", 4.0, (0.9, 0.9, 0.88), bay=3.2, floor=4.0, glass_frac=(0.7, 0.55)),
        "facade_rect": lambda m: windows(m, "facade_rect", 3.0, (1, 1, 1), bay=3.0, floor=3.4),
        "facade_plaster_win": lambda m: windows(m, "plaster", 3.0, (0.95, 0.94, 0.9), bay=3.0, floor=3.0, glass_frac=(0.5, 0.5)),
        "facade_beige_win": lambda m: windows(m, "beige", 3.0, (1, 0.95, 0.88), bay=3.0, floor=3.0, glass_frac=(0.5, 0.5)),
        "facade_shop": lambda m: windows(m, "plaster", 3.0, (0.8, 0.8, 0.82), bay=2.6, floor=3.0, glass_frac=(0.55, 0.5)),
    }
    P.get(name, lambda m: flat(m, (0.6, 0.6, 0.6)))(mat)
    return mat


# ---------------------------------------------------------------- geometry
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
for name in meta["materials"]:
    V = np.load(os.path.join(SCENE, name + ".v.npy")).astype(np.float32)
    F = np.load(os.path.join(SCENE, name + ".f.npy")).astype(np.int32)
    UV = np.load(os.path.join(SCENE, name + ".uv.npy")).astype(np.float32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(F), 3, np.int32))
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", UV[F.ravel()].ravel())
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    me.materials.append(make_material(name))

# ---------------------------------------------------------------- world, light, camera
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
sky = world.node_tree.nodes.new("ShaderNodeTexSky"); sky.sky_type = "NISHITA"
sky.sun_elevation = math.radians(38); sky.sun_rotation = math.radians(150); sky.air_density = 1.4; sky.dust_density = 2.5
world.node_tree.links.new(sky.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 3.2; sun.angle = math.radians(1.0)
so = bpy.data.objects.new("sun", sun); sc.collection.objects.link(so)
so.rotation_euler = (math.radians(52), 0, math.radians(150 + 90))

cam_pts = np.array(meta["camera"])
cam_s = np.array(meta["cam_s"])
cd = bpy.data.cameras.new("cab"); cd.lens = 22; cd.clip_start = 0.3; cd.clip_end = 3000
cam = bpy.data.objects.new("cab", cd); sc.collection.objects.link(cam); sc.camera = cam


def pose(s):
    s = float(np.clip(s, cam_s[0], cam_s[-1]))
    p = np.array([np.interp(s, cam_s, cam_pts[:, j]) for j in range(3)])
    a = np.array([np.interp(min(cam_s[-1], s + 35), cam_s, cam_pts[:, j]) for j in range(3)])
    b = np.array([np.interp(max(cam_s[0], s - 5), cam_s, cam_pts[:, j]) for j in range(3)])
    d = a - b; d[2] = 0; d /= np.linalg.norm(d)
    left = np.array([-d[1], d[0], 0])
    eye = p + left * 0.55 + [0, 0, 2.45]
    tgt = a + left * 0.3 + [0, 0, 1.9]
    v = tgt - eye
    yaw = math.atan2(v[1], v[0]) - math.pi / 2
    pitch = math.atan2(v[2], math.hypot(v[0], v[1]))
    return eye, (math.pi / 2 + pitch, 0, yaw)


def key(frame, s):
    eye, rot = pose(s)
    cam.location = eye; cam.rotation_euler = rot
    cam.keyframe_insert("location", frame=frame); cam.keyframe_insert("rotation_euler", frame=frame)


r = sc.render
r.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = int(opt.get("samples", 12))
sc.cycles.use_denoising = True
sc.cycles.denoiser = "OPENIMAGEDENOISE"
sc.cycles.max_bounces = 3; sc.cycles.diffuse_bounces = 2; sc.cycles.glossy_bounces = 2; sc.cycles.transparent_max_bounces = 2
sc.cycles.use_adaptive_sampling = True
w, h = opt.get("res", (960, 540))
r.resolution_x, r.resolution_y = w, h
r.fps = FPS
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Medium High Contrast"
r.image_settings.file_format = "JPEG"; r.image_settings.quality = 92
sc.render.use_persistent_data = True

if "still" in opt:
    key(1, float(opt["still"]))
    sc.frame_set(1)
    r.filepath = os.path.join(OUT, "still_%05d.jpg" % int(float(opt["still"])))
    bpy.ops.render.render(write_still=True)
else:
    f0, f1 = opt.get("frames", (1, int((cam_s[-1] - cam_s[0]) / SPEED * FPS)))
    for f in range(f0, f1 + 1):
        key(f, cam_s[0] + (f - 1) * SPEED / FPS)
    sc.frame_start, sc.frame_end = f0, f1
    r.filepath = os.path.join(OUT, "f_")
    bpy.ops.render.render(animation=True)
if "save" in opt:
    bpy.ops.wm.save_as_mainfile(filepath=opt["save"])
