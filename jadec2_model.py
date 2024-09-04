"""
Jade Cacoon 2 3D model research and extract script
Credits: Linkz, Zade

Done:
 - Parsing all models
 - Parsing all parts
 - Vertices
 - UVs (texture coordinates)
 - Vertex Normals
 - Tristrip rotation
 - Tristrip/Face generation
 - Texture id for each model
 - Texture extracting
To do:
 - Skeleton (bone hierarchy)
 - Animations


mainly testing on .xsmd
also works on some .xobd files such as acomp.xobd
ps2 sometimes counts sections in rows (rows are 16 bytes long)
make sure to have hex editor showing 16 byte rows.
having 4 byte grouping also helps with finding floats and 32 bit values
example of a row in birda: 12000010 00FFFFFF 00000000 05000000

1 model is made up of multiple parts
1 part forms 1 triangle strip (vertices, uvs, vertex normals and a strip of faces)
"""

import struct as s
import os
import sys
from PIL import Image

# ---------------------
file_path =  "cafu.xsmd" # cafu.xsmd birda0_256.xsmd

export_obj = True # export .obj model file
export_textures = True # export .png texture files
export_path = "out" # folder
export_faces = True
export_mode = 0 # 0: Single .obj | 1: .obj per part | 2: Single .obj w/ separate objects


#print_general = True
print_model = True
print_part = False
print_texture = True
print_export = True
# ---------------------


if len(sys.argv) > 1:
	print("!Arg mode!\n")
	if os.path.exists(sys.argv[1]):
		file_path = sys.argv[1]
	else:
		print(f"'{sys.argv[1]}' doesn't exist\n")
		sys.exit()

file_basename = os.path.basename(file_path)
file_basename = file_basename[:-5] # removes the extension
# print(file_basename)
# sys.exit()

def main():

	models = []
	model_infos = []

	with open(file_path, 'rb') as f:
		unk = r_u32(f) # model count? file type? 4 xsmd, 5 xbmd
		unk = r_u32(f) # size of header?
		off_model = r_u32(f) # offset to first model
		off_textures = r_u32(f) # end of model or offset to textures
		off_unknown = r_u32(f) # animations, collisions, who knows? (also can be used as textures end)

		f.seek(off_model)
		# XVI  - x vertex info? vif?
		# XMDL - x model
		# SKIN - skinning(relation between vertex and bone)/skeleton/animation/morphs
		# NORM - normals?

		f.seek(32, 1)

		unk = r_u32(f)
		num_bones = r_u32(f) # maybe number of bones?????
		unk = r_u32(f)
		num_models = r_u32(f)                # maybe number of models??????
		num_models_again = r_u32(f) # again?
		num_bones_again = r_u32(f)
		unk = r_u32(f)
		unk = r_u32(f)
		unk = r_u32(f)
		unk = r_u32(f)
		unk = r_u32(f)
		unk = r_u32(f)
		num_bones_again2 = r_u32(f)
		unk = r_u32(f)
		unk = r_u32(f)
		num_bones_again3 = r_u32(f)

		if num_models == 0 or num_models > 500:
			print("No models!")
			return None

		for i in range(num_bones): # len 0x50 d80 bones?
			#bone_data_maybe = f.read(0x50)
			unk = r_u32(f) # type?
			unk = r_u32(f)
			unk = r_u32(f)
			unk = r_s16(f)
			unk = r_s16(f) # index?

			unk = r_s16(f) # parent index?
			unk = r_s16(f) # parent index?
			unk = r_u32(f)
			unk = r_u32(f)
			unk = r_u32(f)

			unk_vec3 = r_vec4_to_3(f) # position, rotation?
			rot = r_vec4_to_3(f) # rotation (euler radians)
			f.seek(16, 1)#unk_vec4 = r_vec4(f) # padding?

		# ---
		f.seek(4, 1)
		num_models_again2 = r_u32(f)
		f.seek(8, 1)
		bbox_min = r_vec4_to_3(f) # could be bounding box
		bbox_max = r_vec4_to_3(f)


		f.seek(16, 1) # not always 16 bytes. cafu needs to skip 32 bytes
		if num_models_again2 > 2:
			f.seek(16, 1)

		
		for i in range(num_models_again2): # len 0x60 d96
			#unk_list_2 = f.read(0x60)
			model_info = dict(
				unk = f.read(4),
				unk1 = f.read(4),
				unk2 = f.read(4),
				unk3 = f.read(2),
				texture_index = r_u16(f),
				unk_floats = (r_f32(f), r_f32(f), r_f32(f), r_f32(f), r_f32(f), r_f32(f), r_f32(f)),
				unk4 = f.read(4),
				unk5 = f.read(0x30)
			)
			model_infos.append(model_info)
		# ---

		for m in range(num_models):#num_models):
			if print_model:
				print(f"\nmodel idx: {m} | offset {htell(f)}")

			unk = r_u32(f)
			num_parts = r_u32(f)
			num_total_rows = r_u32(f) # multiply by 16 to get the buffer size
			f.seek(4, 1)

			unk_vec3 = r_vec4_to_3(f)
			unk_vec3 = r_vec4_to_3(f)

			mesh_parts = []
			for i in range(num_parts):
				if print_part:
					print("model idx:", m, "part idx:", i, "start", htell(f))

				# if m > 0:
				# 	return None

				# print("\nPARTTSSSSSSSSSSS")
				# return None

				# parts buffer actually starts here (check if all files have 2 of 08000030...)
				# f.seek(32, 1) # 2 of 08000030
				check_08000030(f)
				check_04000030(f)

				off_current_part = f.tell()
				num_mesh_rows = r_u16(f)
				#len_mesh = num_mesh_rows * 16 + 16
				f.seek(6, 1)
				rotate = True if r_u32(f) == 1 else 0                   # if the faces need to be flipped or not
				f.seek(4, 1)
				f.seek(0x20, 1) # d32
				num_vertices = r_u8(f)
				f.seek(0xf, 1)  # d15
				f.seek(0x20, 1) # d32
				num_vertices = r_u8(f)
				f.seek(0xf, 1)  # d15

				# ---
				f.seek(4, 1)
				vertices = []
				tex_coords = []
				vtx_normals = []

				r_vif_element_header(f) # 0x45 vertices

				for j in range(num_vertices):
					vertices.append(r_vec3(f))

				r_vif_element_header(f) # 0x46 uvs aka texture coordinates

				for j in range(num_vertices):
					tex_coords.append(r_vec2(f))

				r_vif_element_header(f) # 0x47 vertex normals

				for j in range(num_vertices):
					vtx_normals.append(r_vtxnormal(f))
					#print(vtx_normals[j])

				r_vif_element_header(f) # 0x47 # unknown

				for j in range(num_vertices):
					f.seek(4, 1)
				# ---

				f.seek(12, 1) # always 00000017 04040001 00000000

				#print(vertices[0])
				mesh_parts.append((vertices, tex_coords, vtx_normals, rotate))

				if print_part:
					print("reached", htell(f))#phex(f)
					print()

			f.seek(16, 1)

			if print_model:
				print(f"model end {htell(f)}")

			models.append(mesh_parts)

		for i in range(num_bones):
			f.seek(0x90, 1) # lots of floats

		f.seek(16, 1)





		# .TM2 "TIM2" texture archive

		if f.read(4) == b'TIM2':
			print("\nReached textures!")
		else:
			f.seek(off_textures)
			if f.read(4) == b'TIM2':
				print("\nJumped to textures")

		tm2_revision = r_u8(f)
		tm2_format = r_u8(f)
		tm2_num_textures = r_u16(f)
		f.seek(0x78, 1) # d120

		if tm2_format != 1:
			print("New tm2 format!!")
			return None

		textures = []
		images = []

		for i in range(tm2_num_textures): # tm2_num_textures
			print(f"\ntexture {i}", htell(f))
			tex = dict(
				total_size = r_u32(f),
				palette_size = r_u32(f),
				index_size = r_u32(f),
				header_size = r_u16(f),
				num_colors_used = r_u16(f),
				format = r_u8(f),
				num_mipmaps = r_u8(f),
				clut_color_type = r_u8(f),
				color_type = r_u8(f),
				width = r_u16(f),
				height = r_u16(f),
				gs_register_0 = f.read(8),
				gs_register_1 = f.read(8),
				gs_flags_register = f.read(4),
				gs_clut_register  = f.read(4),
				empty = f.read(0x50)
			)

			tex["index_buffer"] = f.read(tex["index_size"])
			tex["palette_buffer"] = f.read(tex["palette_size"])
			#print(tex["index_buffer"][0:128])

			#print(tex)

			if tex["format"] != 0:
				print("New texture format!!")
				return None
			if tex["num_mipmaps"] != 1:
				print("More mipmaps!!")
				return None

			# textures.append(tex)

			print("end", htell(f))


			bits_per_pixel = 8
			# not the correct way. see how other tools do it
			num_bytes_per_color = tex["palette_size"] // 256 # tex["num_colors_used"]


			if export_textures:
				new_pixels = []
				
				new_palette = get_palette_data(tex["palette_buffer"], tex["palette_size"], num_bytes_per_color, False)
				#print(new_palette)

				if bits_per_pixel == 8:
					new_palette = tile_palette(new_palette, 8, 2)
					index_data = list(tex["index_buffer"])
					
					for j, index in enumerate(index_data):
						new_pixels.append(new_palette[index])

				img = Image.new('RGBA', (tex["width"], tex["height"]))
				img.putdata(new_pixels)
				#img.save(f"{file_basename}.{i}.png")

				# img_pal = Image.new('RGBA', (16, 16)) # save palette
				# img.putdata(new_palette)
				# img.save(f"{file_basename}_palette.{i}.png")

				images.append(img)

		print("\n:::", htell(f))





	# EXPORTING


	if export_obj or export_textures:
		if not os.path.isdir(export_path): # if export folder doesn't exist, create it
				os.makedirs(export_path)

	if export_textures:
		for i, img in enumerate(images):
			img.save(f"{export_path}/{file_basename}.{i}.png")


	if export_obj == False: # cancel exporting .obj
		return None


	# overrides for testing
	#export_mode = 0 # 0: Single .obj | 1: .obj per part | 2: Single .obj w/ separate objects
	#export_faces = True

	if export_mode == 0 or export_mode == 2:
		obj_data  = f"# {file_basename}\n"
		obj_data += f"# models: {num_models}\n"

		if export_textures:
			obj_data += f"mtllib {file_basename}.mtl\n"

			mtl_data = f"# material {i}"

			for i in range(tm2_num_textures):
				mtl_data += f"\n\nnewmtl {file_basename}.{i}"
				mtl_data += mat_info_preset
				mtl_data += f"\nmap_Kd {file_basename}.{i}.png"

	current_max = 0

	for m, model in enumerate(models):

		if export_mode == 2:
			obj_data += f"o {file_basename}.{m}\n"
			obj_data += f"g {file_basename}.{m}\n"

		for p, part in enumerate(model):
			if print_export:
				# print(f"writing part {p+1}")
				pass

			if export_mode == 1:
				obj_data = ""

			vertices = part[0]
			tex_coords = part[1]
			vtx_normals = part[2]
			rotate = part[3]

			# print(len(vertices))
			# y and z negative to adjust orientation

			obj_data += "\n"

			for v in vertices:
				obj_data += f"v {v[0]} {-v[1]} {-v[2]}\n" # vertices. 
			obj_data += "\n"

			for vt in tex_coords:
				obj_data += f"vt {vt[0]} {-vt[1] + 1}\n" # temp uvs
			obj_data += "\n"

			for vn in vtx_normals:
				obj_data += f"vn {vn[0]} {-vn[1]} {-vn[2]}\n" # vertex normals
			obj_data += "\n"

			if export_textures:
				texture_index = model_infos[m]["texture_index"]
				obj_data += f"\nusemtl {file_basename}.{texture_index}\n"

			if export_faces: # faces / triangles / polygons
				initial_strip = [j for j in range(current_max, current_max + len(vertices))]
				#print(initial_strip)
				current_max = initial_strip[-1] + 1
				#print(current_max)

				new_faces = tristrip_to_faces(initial_strip, rotate)

				for f in new_faces:
					obj_data += f"f {f[0] + 1}/{f[0] + 1}/{f[0] + 1} {f[1] + 1}/{f[1] + 1}/{f[1] + 1} {f[2] + 1}/{f[2] + 1}/{f[2] + 1}\n"

				# if p == 1:
				# 	break

			if export_mode == 1:
				export_final_path = f"{export_path}/{file_basename}.{p}.obj"
				with open(export_final_path, 'w') as out_f:
					out_f.write(obj_data)
				if print_export:
					print("Exported", export_final_path)

	if export_mode == 0 or export_mode == 2:

		export_final_path = export_path + "/" + file_basename

		with open(export_final_path + ".obj", "w") as out_f:
			out_f.write(obj_data)

		if export_textures:
			with open(export_final_path + ".mtl", "w") as out_f:
				out_f.write(mtl_data)

		if print_export:
			print("Exported ", export_final_path)


	print("\nExporting Done")

mat_info_preset = """
Ns 0.000000
Ka 1.000000 1.000000 1.000000
Kd 0.800000 0.800000 0.800000
Ks 0.000000 0.000000 0.000000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 1"""

def get_palette_data(palette_buffer, palette_size, num_bytes, alpha):
	colors = []
	if num_bytes == 3: # PALETTE_FORMAT_RGB888

		for i in range(0, len(palette_buffer), 3):
			r,g,b,a = int(palette_buffer[i]), int(palette_buffer[i+1]), int(palette_buffer[i+2]), 255
			colors.append((r,g,b,a))

		# for i in range(0, len(palette_buffer), 4):
		# 	r,g,b,a = (int(palette_buffer[i]), int(palette_buffer[i+1]), int(palette_buffer[i+2]), int(palette_buffer[i+3]))
		# 	colors.append((r,g,b,a))

	elif num_bytes == 4: # PALETTE_FORMAT_RGBA8888
		for i in range(0, palette_size, 4):
			r,g,b,a = int(palette_buffer[i]), int(palette_buffer[i+1]), int(palette_buffer[i+2]), int(palette_buffer[i+3])
			if int(palette_buffer[-1]) <= 0x80:
				a = int(float(0xff) * (float(a) / float(0x80)))
			colors.append((r,g,b,a))


	# if alpha == True:
	#     (r0,g0,b0,a0) = colors[0]
	#     colors[0] = (r0,g0,b0,0)
	return colors

def tile_palette(palette, tile_x, tile_y):
	ntx = 16 // tile_x
	nty = 16 // tile_y
	i = 0
	new_palette = [(0,0,0,0) for j in range(256)]

	for ty in range(nty):
		for tx in range(ntx):
			for y in range(tile_y):
				for x in range(tile_x):
					new_palette[(ty * tile_y + y) * 16 + (tx * tile_x + x)] = palette[i]
					i += 1
	return new_palette

def tristrip_to_faces(strip, rotate):
	#rotate = True # clockwise/anticlockwise rotation
	face_group = []
	for i in range(len(strip) - 2):
		if rotate:
			face = (strip[i+2], strip[i+1], strip[i+0])
		else:
			face = (strip[i], strip[i+1], strip[i+2])

		rotate = not rotate
		face_group.append(face)
	return face_group

def check_08000030(f):
	if f.read(8) == b'\x08\x00\x00\x30\x02\x00\x00\x00':
		f.seek(8, 1) # skip the rest of the row
		while True:
			if f.read(8) == b'\x08\x00\x00\x30\x02\x00\x00\x00':
				f.seek(8, 1) # skip the rest of the row
			else:
				f.seek(-8, 1) # go back 8 bytes
				break
	else:
		f.seek(-8, 1) # go back 8 bytes

def check_04000030(f):
	if f.read(8) == b'\x04\x00\x00\x30\x02\x00\x00\x00':
		f.seek(8, 1) # skip the rest of the row
		while True:
			if f.read(8) == b'\x04\x00\x00\x30\x02\x00\x00\x00':
				f.seek(8, 1) # skip the rest of the row
			else:
				f.seek(-8, 1) # go back 8 bytes
				break
	else:
		f.seek(-8, 1) # go back 8 bytes

def htell(f):
	#return hex(f.tell())
	return str(hex(f.tell()))[2:]
def phex(f): # print current address as hex
	#print(str(htell(f))[2:])
	print(str(htell(f)))

def r_vtxnormal(f):
	x = s.unpack("h", f.read(2))[0]
	y = s.unpack("h", f.read(2))[0]
	z = s.unpack("h", f.read(2))[0]
	f.seek(2, 1)
	return (x / 32767, y / 32767, z / 32767) # not sure what the divisor should be

def r_vif_element_header(f):
	e_type = r_u8(f)
	e_unk = r_u8(f)
	e_count = r_u8(f)
	e_unk = r_u8(f)


def r_u8(f):
	return s.unpack("B", f.read(1))[0]
def r_s16(f):
	return s.unpack("h", f.read(2))[0]
def r_u16(f):
	return s.unpack("H", f.read(2))[0]
def r_s32(f):
	return s.unpack("i", f.read(4))[0]
def r_u32(f):
	return s.unpack("I", f.read(4))[0]
def r_f32(f):
	return s.unpack("f", f.read(4))[0]
def r_vec2(f):
	return s.unpack("ff", f.read(8))
def r_vec3(f):
	return s.unpack("fff", f.read(12))
def r_vec4(f):
	return s.unpack("ffff", f.read(16))
def r_vec4_to_3(f):
	return s.unpack("ffff", f.read(16))[:3]

main()
