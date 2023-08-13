# Graphics exploration with vulkan

```sh
cargo run --release --bin dbg -- --gltf-file ../glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf --rsf-file data/ambient_light.ron.rsf --rsf-file data/spot_light.ron.rsf
```

Graphics experimentation with vulkan (mostly contained in trekanten/).

Currently supports metallic-roughness PBR as defined by gltf and also shadow mapping for spot & directional lights (WIP).

Free flying camera controls:

* WASD to move.
* Mouse left button drag to look around.
* ESC to pause/resume.

Rendering controls:

* R to reload all shaders

Renders:

* [x] Box
* [x] BoxTextured
* [x] BoxVertexColors
* [x] Cube
* [x] Sponza

from <https://github.com/KhronosGroup/glTF-Sample-Models>

## Notes

Two main libs: ramneryd & trekanten. Binary `dbg` is used to run the "engine".

### ramneryd

Prototype for integrating specs ECS with trekanten vulkan wrapper.

* Transforms are uploaded with push constants.
* Supports two materials: PBR & unlit.
* WIP support for scenes with data/*.ron.rsf files. These can be loaded to get different light conditions.

#### Descriptor set organisation

Descriptor set 0 holds the "engine" data: light matrices, shadow matrices, main camera view data etc, shadow maps.
Descriptor set 1 is material-specific.

### trekanten

Initially, the idea was for this to be (yet another) low-level graphics lib wrapping vulkan but with convenient API. I
also had the idea that I might support WPGU as a separate backend to run in the browser. As time has passed, I've realized
that I mostly want to learn how rendering works rather than abstracting backends so vulkan has leaked a bit more and more
out of this lib, which I think is fine. It might move towards becoming less of a wrapper and more of a collection of utils
when working with vulkan.

### Useful links

Vulkan tutorial: <https://vulkan-tutorial.com/>
Vulkano: <https://github.com/vulkano-rs/vulkano>

Vulkan coordinate system: <https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/>

Opengl modern: <http://in2gpu.com/opengl-3/>, <http://in2gpu.com/2016/02/26/opengl-fps-camera/>

Derivation of matrix for normals:
<http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/>

PBR:
<https://github.com/moneimne/glTF-Tutorials/tree/master/PBR>
<https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#appendix-b-brdf-implementation>

gltf sample viewer:
<https://github.com/KhronosGroup/glTF-Sample-Viewer/>
Includes shader sources and some explanation

Good explanation, mostly from an artists point of view
<https://academy.substance3d.com/courses/the-pbr-guide-part-1>
<https://academy.substance3d.com/courses/the-pbr-guide-part-2>

gltf:
<https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/README.md>
