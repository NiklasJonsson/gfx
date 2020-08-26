#! /usr/bin/env sh

usage() {
	echo "Usage: $0 <shader-dir>"
}

if [ $# -ne 1 ]; then
	echo "Error missing arg"
	usage
	exit 1
fi

shader_dir="$1"

which glslc > /dev/null
if [ $? -ne 0 ]; then
	echo "Error: Can't find glslc"
	exit 1
fi

glslc -fshader-stage=vert "$shader_dir"/shader.vert.glsl -o "$shader_dir"/vert.spv
glslc -fshader-stage=frag "$shader_dir"/shader.frag.glsl -o "$shader_dir"/frag.spv
