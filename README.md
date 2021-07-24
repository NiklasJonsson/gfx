# ramneryd

A barebones game framework, currently: input, imgui ui and rendering. Rendering is built on trekanten, a vulkan wrapper.
Currently supports metallic-roughness PBR as defined by gltf and also shadow mapping for spot & directional lights (WIP).

Free flying camera controls:
* WASD to move.
* Mouse left button drag to look around.
* ESC to pause/resume.

Rendering controls:
* R to reload all shaders

Renders:

- [x] Box
- [x] BoxTextured
- [x] BoxVertexColors
- [x] Cube
- [x] Sponza

from https://github.com/KhronosGroup/glTF-Sample-Models
