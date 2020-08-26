C:\Users\jonss\tools\shaderc\bin\glslc.exe -fshader-stage=vert uniform_color_frag.glsl -o fs_uniform_color.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -fshader-stage=vert pos_only_vert.glsl -o vs_pos_only.spv

# Vert
C:\Users\jonss\tools\shaderc\bin\glslc.exe -fshader-stage=vert pbr_gltf_vert.glsl -o vs_pbr_base.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -DHAS_TEX_COORDS=1 -DTEX_COORDS_LOC=2 -fshader-stage=vert pbr_gltf_vert.glsl -o vs_pbr_uv.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -DHAS_TEX_COORDS=1 -DTEX_COORDS_LOC=2 -DHAS_VERTEX_COLOR=1 -DVCOL_LOC=3 -fshader-stage=vert pbr_gltf_vert.glsl -o vs_pbr_uv_vcol.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -DHAS_TEX_COORDS=1 -DTEX_COORDS_LOC=2 -DHAS_TANGENTS=1 -DTAN_LOC=3 -DBITAN_LOC=4 -fshader-stage=vert pbr_gltf_vert.glsl -o vs_pbr_uv_tan.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -DHAS_TEX_COORDS=1 -DTEX_COORDS_LOC=2 -DHAS_VERTEX_COLOR=1 -DVCOL_LOC=3 -DHAS_TANGENTS=1 -DTAN_LOC=4 -DBITAN_LOC=5 -fshader-stage=vert pbr_gltf_vert.glsl -o vs_pbr_uv_vcol_tan.spv

# Frag
C:\Users\jonss\tools\shaderc\bin\glslc.exe -fshader-stage=frag pbr_gltf_frag.glsl -o fs_pbr_base.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -fshader-stage=frag -DHAS_TEX_COORDS=1 -DTEX_COORDS_LOC=2 -DHAS_BASE_COLOR_TEXTURE=1 pbr_gltf_frag.glsl -o fs_pbr_bc_tex.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -fshader-stage=frag -DHAS_TEX_COORDS=1 -DHAS_TANGENTS=1 -DTEX_COORDS_LOC=2 -DTAN_LOC=3 -DBITAN_LOC=4 -DHAS_BASE_COLOR_TEXTURE=1 -DHAS_METALLIC_ROUGHNESS_TEXTURE=1 -DHAS_NORMAL_MAP=1 pbr_gltf_frag.glsl -o fs_pbr_bc_mr_nm_tex.spv
C:\Users\jonss\tools\shaderc\bin\glslc.exe -fshader-stage=frag -DHAS_TEX_COORDS=1 -DTEX_COORDS_LOC=2 -DHAS_VERTEX_COLOR=1 -DVCOL_LOC=3 pbr_gltf_frag.glsl -o fs_pbr_uv_vcol.spv