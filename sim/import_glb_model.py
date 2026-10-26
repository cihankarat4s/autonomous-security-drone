"""Turn a downloaded glTF model into a static Gazebo model.

Sketchfab models are Y-up, in arbitrary units, often skinned and with large
textures. This bakes the bind pose into one static mesh, rotates it to
Gazebo's Z-up, scales it to a real-world length, puts it on the ground at
the origin facing +x, shrinks the base colour texture, and writes
sim/models/<name>/ (model.config, model.sdf, meshes/<name>.glb).

Needs trimesh and Pillow (not project dependencies; a scratch venv is fine):

    python sim/import_glb_model.py ~/datasets/assets/labrador_dog/scene.gltf \
        --name labrador_dog --length 1.05 --credit "..."
"""
import argparse
import io
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

SIM = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--length", type=float, required=True,
                        help="real length of the longest horizontal side, metres")
    parser.add_argument("--texture", type=int, default=1024, help="max base colour size, px")
    parser.add_argument("--credit", required=True, help="attribution line for model.config")
    args = parser.parse_args()

    scene = trimesh.load(args.source)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene

    # Y-up -> Z-up: rotate +90 degrees about x.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    # Longest horizontal side along +x.
    ext = mesh.extents
    if ext[1] > ext[0]:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
    mesh.apply_scale(args.length / max(mesh.extents[:2]))
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is not None:
        image = image.convert("RGB")
        image.thumbnail((args.texture, args.texture))
        mesh.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.9)

    out = SIM / "models" / args.name
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    glb = out / "meshes" / f"{args.name}.glb"
    glb.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh)))
    size = mesh.extents
    (out / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{args.name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    {args.credit}
    Converted by sim/import_glb_model.py: static bind pose, Z-up, scaled to
    {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, base colour {args.texture} px.
  </description>
</model>
""")
    (out / "model.sdf").write_text(f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="{args.name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <mesh><uri>meshes/{args.name}.glb</uri></mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
""")
    print(f"Saved {out}: {len(mesh.faces)} faces, {size.round(2).tolist()} m, "
          f"{glb.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
