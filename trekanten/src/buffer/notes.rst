Buffer management
*****************

There are a few design features we want to acheive with buffers:

*Mutability management*
  The use be able to specify if the buffer is mutable or immutable. For mutable buffers, the device
  buffer should be accessible from both the gpu and cpu, persistently mapped. Furthermore, it is
  double-buffered to allow modyfing data while still drawing the previous frame. An immutable buffer
  is single-buffered and is uploaded via staging buffer to device-only memory which should be faster
  to access. It is a runtime error to try to modify a immutable buffer. Otherwise the two should be
  interchangeable. Mutable buffers should only be mutated between a next_frame() and submit_frame()
  call, as we won't otherwise know which frame idx to use.

*Handle-based device buffer management*
  References to gpu-api managed resources are passed around with handles to ensure lifetime are
  proper. E.g. that they are not destroyed before they are used on the gpu.

*Buffers are typed according to their usage*
  Vertex, index and uniform buffers have different types.

*Provide convenience type and ergonomic api*
  Provide both host-only buffers (frontend-only) for convenience of managing buffer data and allow
  creating buffer descriptors from them easily.