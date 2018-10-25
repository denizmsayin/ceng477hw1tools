from argparse import ArgumentParser
from copy import deepcopy
import xml.etree.cElementTree as ET

import numpy as np
import cv2

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'xml_path',
        help='path to the xml scene file')
    parser.add_argument(
        'image_path',
        help='path to the image to be inserted as background')
    parser.add_argument(
        'output_xml_path',
        help='path to the output xml scene file')
    parser.add_argument(
        '--t-ratio-multiplier', '-tm',
        help='value by which to scale the distance of the furthest image plane. 0 '
             'means camera position, 1 means the furthest vertex, 2 means twice as '
             'far as the furthest vertex etc., default is 1.1',
        type=float,
        default=1.1)
    parser.add_argument(
        '--ambient-multiplier', '-am',
        help='value by which to multiply ambient reflection coefficients, default is 1.0',
        type=float,
        default=1.0)
    parser.add_argument(
        '--diffuse-multiplier', '-dm',
        help='value by which to multiply diffuse reflection coefficients, default is 1.0',
        type=float,
        default=1.0)
    parser.add_argument(
        '--specular-multiplier', '-sm',
        help='value by which to multiply specular reflection coefficients, default is 1.0',
        type=float,
        default=1.0)
    parser.add_argument(
        '--mirror-multiplier', '-mm',
        help='value by which to multiply mirror reflection coefficients, default is 1.0',
        type=float,
        default=1.0)
    parser.add_argument(
        '--phong-exponent', '-pe',
        help='phong exponent for specular reflection, default is 1',
        type=int,
        default=1)
    args = parser.parse_args()
    if args.xml_path == args.output_xml_path:
        raise ValueError('Input and output paths should not be the same')
    return args

def parse_scalar(text):
    try:
        scalar = float(text.strip())
        return scalar
    except Exception:
        raise ValueError('Could not parse scalar from "{}"'.format(text))

def parse_vec(text):
    try:
        vec = np.array([float(token) for token in text.strip().split(' ')])
        return vec
    except Exception:
        raise ValueError('Could not parse vec from "{}"'.format(text))

def parse_mat(text):
    try:
        lines = text.strip().split('\n')
        vecs = [parse_vec(line) for line in lines]
        mat = np.vstack(vecs)
        return mat
    except Exception:
        raise ValueError('Could not parse mat3 from "{}"'.format(text))

def make_unit(vec):
    return vec / np.linalg.norm(vec)

def parse_camera(camera_element):
    camera = {}
    camera['position'] = parse_vec(camera_element.find('Position').text)
    camera['gaze'] = make_unit(parse_vec(camera_element.find('Gaze').text))
    camera['up'] = make_unit(parse_vec(camera_element.find('Up').text))
    camera['u'] = np.cross(camera['gaze'], camera['up'])
    camera['near_plane'] = parse_vec(camera_element.find('NearPlane').text)
    camera['near_distance'] = parse_scalar(camera_element.find('NearDistance').text)
    camera['mid'] = camera['position'] + camera['gaze'] * camera['near_distance']
    return camera

def get_implane_corners(camera):
    m, u, v = camera['mid'], camera['up'], camera['u']
    l, r, b, t = camera['near_plane']
    tl = m + l * v + t * u
    br = m + r * v + b * u
    return tl, br

def solve_implane(camera, furthest_point):
    # constants are marked as (*)
    # image plane equation formulation:
    # p(u, v) = m + (alpha) * u + (beta) * v
    # ray to furthest point equation formulation
    # r(t) = o + (t) * d
    m, u, v = camera['mid'], camera['up'], camera['u']
    l, r, b, t = camera['near_plane']
    o = camera['position']
    d = make_unit(furthest_point - o)
    lhs = np.vstack((u, v, -d)).T
    rhs = o - m
    alpha, beta, t = np.linalg.solve(lhs, rhs)
    return alpha, beta, t

def translate_point(origin, point, translation_ratio):
    d = point - origin
    return origin + d * translation_ratio

def generate_triangles(img, tl, u, v, offset):
    # u down, v right
    float_image = img.astype('float64') / 255
    h, w = img.shape[0], img.shape[1]
    vertices = [tl + i*v + j*u for j in range(h+1) for i in range(w+1)]
    triangles = []
    colors = []
    # multiprocess faster map
    for j in range(h):
        c = j * (w + 1)
        for i in range(w):
            # ccw vertex ordering
            face1 = (offset+c+i+1, offset+c+i+w+2, offset+c+i+2)
            face2 = (offset+c+i+2, offset+c+i+w+2, offset+c+i+w+3)
            triangles.append((face1, face2))
            colors.append(float_image[j][i])
    return vertices, triangles, colors

# get the command line arguments
args = parse_arguments()
# parse the XML file
tree = ET.parse(args.xml_path)
root = tree.getroot()
# parse the camera position and vertex data into ndarrays
camera = parse_camera(root.find('Cameras').find('Camera'))
vertices = parse_mat(root.find('VertexData').text)
# find the furthest point from the camera
dists = np.sum((camera['position'] - vertices)**2, axis=1)
furthest_point = vertices[np.argmax(dists), :]
# find 't' for the furthest point wrt. the camera pos.
furthest_t = np.linalg.norm(furthest_point - camera['position'])
# find the image plane intersection for the ray going to the furthest point
alpha, beta, implane_t = solve_implane(camera, furthest_point)
t_ratio = furthest_t / implane_t * args.t_ratio_multiplier
# get the corners of the image plane
tl, br = get_implane_corners(camera)
# translate the corners for image projection
im_tl = translate_point(camera['position'], tl, t_ratio)
im_br = translate_point(camera['position'], br, t_ratio)
iml, imr, imb, imt = t_ratio * camera['near_plane']
# now we can use the image generator
img = cv2.cvtColor(cv2.imread(args.image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
h, w = img.shape[0], img.shape[1]
u = camera['up'] * (imb - imt) / h
v = camera['u'] * (imr - iml) / w
vertices, triangles, colors = generate_triangles(img, im_tl, u, v, vertices.shape[0])
# now, time to add the new data to the xml
fmt_str = '{:.6f} {:.6f} {:.6f}'
# insert the new vertices
vertices_text = [fmt_str.format(*vertex) for vertex in vertices]
root.find('VertexData').text += '\n' + '\n'.join(vertices_text) + '\n'
# insert the new materials
phong_exponent_str = str(args.phong_exponent)
materials_xml = root.find('Materials')
nmats = len(materials_xml.findall('Material'))
material_template = deepcopy(materials_xml.find('Material'))
for i, color in enumerate(colors, 1):
    material = ET.SubElement(materials_xml, 'Material', attrib={'id':str(nmats+i)})
    ambient_reflectance = ET.SubElement(material, 'AmbientReflectance')
    diffuse_reflectance = ET.SubElement(material, 'DiffuseReflectance')
    specular_reflectance = ET.SubElement(material, 'SpecularReflectance')
    mirror_reflectance = ET.SubElement(material, 'MirrorReflectance')
    phong_exponent = ET.SubElement(material, 'PhongExponent')
    ambient_reflectance.text = fmt_str.format(*(color * args.ambient_multiplier))
    diffuse_reflectance.text = fmt_str.format(*(color * args.diffuse_multiplier))
    specular_reflectance.text = fmt_str.format(*(color * args.specular_multiplier))
    mirror_reflectance.text = fmt_str.format(*(color * args.mirror_multiplier))
    phong_exponent.text = phong_exponent_str
# insert meshes
objects_xml = root.find('Objects')
nmeshes = len(objects_xml.findall('Mesh'))
# there may not be any prior meshes in the scene, no ez life template :(
fmt_str = '{} {} {}'
for i, triangle_group in enumerate(triangles, 1):
    mesh = ET.SubElement(objects_xml, 'Mesh', attrib={'id':str(nmeshes+i)})
    material = ET.SubElement(mesh, 'Material')
    material.text = str(nmats + i)
    faces = ET.SubElement(mesh, 'Faces')
    triangle_group_text = [fmt_str.format(*triangle) for triangle in triangle_group]
    faces.text = '\n' + '\n'.join(triangle_group_text) + '\n'
tree.write(args.output_xml_path)
