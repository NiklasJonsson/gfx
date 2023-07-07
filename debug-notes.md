# Debug notes

## The blocky directional shadows

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
