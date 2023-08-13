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

### Debugging the lights not showing up

The scene is dark because no light hits the objects, it seems like. Why is light not hitting the objects?

Solution: Make sure to write LightingData::n_lights.

### Debugging the shadow flickering for spot light

It seems like the matrix indices are wrong.

Solution: Well yes, they were very wrong.

### Shadow map for direction light is from incorrect angle

The shadow map for directional light looks like the light is aligned with the flat plane that is the ground.

Solution: Make sure to not forget the view matrix...

### Refactoring is done

It now works like:

1. `light::shadow_pass()`

    1. Build a buffer of view projection matrices for shadow lights.
    2. Write the buffer to the gpu with a blocking call, this is the buffer that each separate shadow pass will bind one
    instance of for the vertex shader view_proj.
    3. Encode depth-only render passes, one for each of the shadow lights, for all entities.
    4. Each of the shadow lights get a `ShadowMap` associated with it that holds indices into the shadow maps and into the
    matrices for a specific shadow pass. It also holds a `ShadowType` enum,
    so that the main rendering pass can pass that info on to the fragment shader for texture
    lookups.

2. `light::write_lighting_data`

    This function writes the `LightingData` and `ShadowData` uniforms that needs to be written before the main lit
    render pass.

3. Main render pass
  This uses the LightingData and the ShadowData to run the PBR shaders.

## Shader recompilation and failure management

Goals:

1. Bad shaders shouldn't crash the app on startup.
2. Would be nice with a CLI tool to compile all shaders in the repo
3. Live reload of shaders with manual action

### The Work

For 1 the tricky thing for this is that the dummy pipelines are part of the render initialization and there is not a good
way (currently) to handle them failing to compile graceful and still continue the rendering loop.

For 2, added `check-shaders` that runs through all shaders and compiles them, emitting errors as it goes.

For 3, The current (non-working) solution is to tag all entities that should have its material recomputed and effectively
redo the entire creation. Setting aside the buggy behaviour, this might miss shaders that are not attached to entities,
e.g. the shadow pass shaders, which won't be recompiled. A future solution might look like:

1. The UI (keybind or button) generates recompile event. It could also be coming from a file watcher.
2. This is intercepted somewhere were we have access to the renderer and we recreate all the pipelines with shaders.

This means we have to be able to recreate shaders easily.

* We'd need to register all `GraphicsPipelineDescriptors` that are used.
* We might need to `WaitDeviceIdle` before replacing all the pipelines, to ensure none are in use.
* We'd go through all the shaders (pipelines descriptors?) that are known, get the matching pipeline idx in the
resources storage and recreate it. On subsequent uses, all rendering would use the new pipeline.
* Currently, the pipeline descriptors only accept raw spirv so we'd need to associate the shader paths with the
pipeline descriptor.
* This would not support shaders that are not loaded from file unless we take a function that provides the source? This
might be nice anyhow to make sure it is agnostic.
* The recompilation system could live at the start of `draw_frame` as a manual call.
