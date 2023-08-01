# Notes

## Debugging the blocky directional shadows

**Issue**: The directional shadow maps produce blocky shadows for small objects. This is especially visible when the
light is moving as the pixelated values flicker very cleary.

**Repro**: `cargo run -- --rsf-file data\ambient_light.ron.rsf --spawn-cube --spawn-plane`

The process of directional shadow mapping is largely: We build the bounds of what the light space is, by taking the
camera viewing volume and the scene into account. This results in an AABB in world space that we create an ortographic
projection for and render the shadow into. The shadow rendering pass is largely a matrix * vertex for the vertex shader
and then only a depth write for the fragment. No fragment shader is run and no pixels are written.

This is then read as a texture attachment in the fragment shader for the PBR to determine if the fragment that we are
currently rendering is occluded from the light. Each vertex gets coordinates for each light it might be interacting
with, which are interpolated for the fragments. To check if a fragment is occluded, its shadow coordinates are used
to lookup the depth in the texture for that light. If its z value is larger than the depth, it means that there is
something occluding it in the direction of the light and it should be shadowed. This is used to affect the shading
of that fragment by modifying the contribution of that lights color for the final color of the pixel.

* For the plane and cube scene with a stable directional light, the plane is quite large in the shadow map (in render do)
 and the small cube doesn't really show up. This seems to indicate that the bounds are too large.
* Dumping the bounds in the UI, the bounding volumes seem to be OK? Pixel density for shadow map should be around
33 p/m^2 which is OK? Maybe this is low?

The shadow map in render doc seems to strongly indicate that the map is too large anyhow and this is why the box shadow
is pixelated.

> One possible solution might be to exclude "the ground" from the shadow computations. It could be possible to introduce
a "non shadow caster"/light passthrough tag that we can attach to the "ground" plane to reduce the size of the shadow
volume. Still though, it seems large and that it should be possible to reduce it.

This is how the shadow map volume creation works:

1. Compute the bounds of the view we want to cast shadows in.

    1. Find the camera that is the shadow viewer (the loop exits after the first).
    2. Compute the view matrix (the conversion from world to camera space).
    3. Compute the OOBB of the camera (in camera space).
    4. Invert the view matrix and use it to convert the OOBB to camera space.

2. For each view, this bounding box (OOBB) is converted to the coordinate system of that light.
3. Convert the OOBB to an AABB, that is constructed in a way to minimize artifacts from lights and camera movement.
4. This AABB is used to construct the ortographic projection matrix.

Reading the Microsoft guide to shadow maps, it seems fairly likely that the problem is "perspective aliasing".

Solution: Implement Cascaded shadow maps.

## Refactor the light pass

**Issue**: The light_and_shadow_pass has a bit too many responsibilities.

It would be nice to separate the shadow passes into standalone passes that pass on their information.

Conceptually, I think this is the problem space:

### The shadow render passes

1. Find the lights that cast shadows
2. For each of these:
    1. compute a projection matrix:
        * This is both used in the shadow pass rendering vertex shader to transform the vertices
        * And in the vertex shader for the main render pass to compute the shadow coords for each vertex.
    2. Update the uniform buffer for the shadow render passes
    3. Encode the shadow render pass in the command buffer. This only runs a vertex shader that computes the depths and
    then writes those to a depth buffer.

The output of the shadow render passes is effectively a list of texture and matrices that should be used in the main
render pass.

### The main lighting pass

1. Find the list of lights that affect the scene.
2. Compute PackedLight for all of these. These hold an index into the buffer of shadow matrices and shadow textures
(NOTE: These are not the same)
3. Write PackedLight uniform
4. bind the textures
5. Bind the shadow coords
6. Draw all entities

## Solution

It now works like:

1. `light::shadow_pass()`

    1. Build a buffer of view projection matrices for shadow lights.
    2. Write the buffer to the gpu with a blocking call
    3. Encode depth-only render passes, one for each of the shadow lights, for all entities.
    4. Each of the shadow lights get a `ShadowMap` associated with it that holds a handle to a sub-buffer of (1).
       This is the view projection matric for a specific light. It also holds a `ShadowType` enum,
       so that the main rendering pass can pass that info on to the fragment shader for texture
       lookups.
    5. Also returns the shadow view projection matrices with a buffer handle.

2. `light::write_lighting_data`

    This function writes the `LightingData` uniform that needs to be written before the main lit render pass.

3. Main render pass
  This uses the LightingData and the