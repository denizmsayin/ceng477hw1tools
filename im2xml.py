from argparse import ArgumentParser

import numpy as np
import cv2

class XMLContext:
    XMLDepth = 0
    def __init__(self, tag, attrib=''):
        self.tag = tag
        self.depth = XMLContext.XMLDepth 
        self.attrib = attrib
    def __enter__(self):
        XMLContext.XMLDepth += 1
        print('    ' * self.depth + '<{}{}>'.format(self.tag, self.attrib))
        return self
    def __exit__(self, *args):
        XMLContext.XMLDepth -= 1
        print('    ' * self.depth + '</{}>'.format(self.tag))
    def put(self, *args):
        print('    ' * (self.depth + 1), end='')
        print(*args)

    def putline(tag, *args):
        print('    ' * XMLContext.XMLDepth, end='')
        print('<{}>'.format(tag), end='')
        print(*args, end='')
        print('</{}>'.format(tag))

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'image_path',
        help='path to the image to be triangularized')
    parser.add_argument(
        '--scale', '-s',
        help='factor by which to scale the image, by default each pixel is 1x1',
        type=float,
        default=1.0)
    args = parser.parse_args()
    return args

def generate_triangles(img, scale):
    float_image = img.astype('float64') / 255
    h, w = img.shape[0], img.shape[1]
    vertices = ['{} {} {}'.format(i*scale, -j*scale, 0.0) for j in range(h+1) for i in range(w+1)]
    triangles = []
    colors = []
    for j in range(h):
        c = j * (w + 1)
        for i in range(w):
            face1 = '{} {} {}'.format(c+i+1, c+i+2, c+i+w+2)
            face2 = '{} {} {}'.format(c+i+2, c+i+w+2, c+i+w+3)
            triangles.append((face1, face2))
            r, g, b = float_image[j][i]
            colors.append('{:.6f} {:.6f} {:.6f}'.format(r,g,b))
    return vertices, triangles, colors

def generate_xml(args):
    img = cv2.cvtColor(cv2.imread(args.image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    scale = args.scale
    vertices, triangles, colors = generate_triangles(img, scale)
    h, w = img.shape[0], img.shape[1]
    mid_vertex = (w/2, h/2, 0.0)
    I = 25 * (scale * max(h,w))**2
    d = 5
    points = ((0,0,d),(w*scale,0,d),(0,-h*scale,d),(w*scale,-h*scale,d),(w*scale/2,-h*scale/2,d))
    with XMLContext('Scene'):
        XMLContext.putline('BackgroundColor', '0 0 0')
        print()
        XMLContext.putline('ShadowRayEpsilon', '1e-3')
        print()
        XMLContext.putline('MaxRecursionDepth', '6')
        print()
        with XMLContext('Cameras'):
            with XMLContext('Camera', ' id="1"'):
                XMLContext.putline('Position', w/2, -h/2, 10)
                XMLContext.putline('Gaze', '0 0 -1')
                XMLContext.putline('Up', '0 1 0')
                XMLContext.putline('NearPlane', -w*scale/2, w*scale/2, -h*scale/2, h*scale/2)
                XMLContext.putline('NearDistance', 8)
                XMLContext.putline('ImageResolution', 5*w, 5*h)
                XMLContext.putline('NumSamples', 1)
                XMLContext.putline('ImageName', 'hello.ppm')
        print()
        with XMLContext('Lights'):
            XMLContext.putline('AmbientLight', 25, 25, 25)
            for i, point in enumerate(points, 1):
                with XMLContext('PointLight', ' id="{}"'.format(i)):
                    XMLContext.putline('Position', *point)
                    XMLContext.putline('Intensity', I, I, I)
        print()
        with XMLContext('Materials'):
            for i, color in enumerate(colors, 1):
                with XMLContext('Material', ' id="{}"'.format(i)):
                    XMLContext.putline('AmbientReflectance', color)
                    XMLContext.putline('DiffuseReflectance', color)
                    XMLContext.putline('SpecularReflectance', color)
                    XMLContext.putline('MirrorReflectance', color)
                    XMLContext.putline('PhongExponent', 1)
        print()
        with XMLContext('VertexData') as vertex_data:
            for vertex in vertices:
                vertex_data.put(vertex)
        print()
        with XMLContext('Objects'):
            for i, triangle in enumerate(triangles, 1):
                with XMLContext('Mesh', ' id="{}"'.format(i)):
                    XMLContext.putline('Material', i)
                    with XMLContext('Faces') as faces:
                        faces.put(triangles[i-1][0])
                        faces.put(triangles[i-1][1])

args = parse_arguments()
generate_xml(args)
