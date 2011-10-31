"""
This file provides methods to extract line segments from an image and also to
write such segments as walls in a simpedia-readable xml format

This script takes an image, performs edge detection followed by a Probabilistic
Hough Transform to find straight lines.  It then creates a simpedia <obstacles>
element that contains each of these lines as a wall.

The transform params were cribed from 
http://scikits-image.org/docs/0.3/auto_examples/plot_hough_transform.html
(hence there is potentially of room for improvement)

Additional functionality: there are now options for applying line-thinning
to correct the thickness of walls.

@author: tom.temple@vecna.com
11/24/2011

"""

usage = """
This script takes an image, performs edge detection followed by a Probabilistic
Hough Transform to find straight lines.  It then creates a simpedia <obstacles>
element that contains each of these lines as a wall.

If the -y argument is omitted it will try changing the -f argument's file 
extension to .yaml.

If the -s FILE.EXT argument is used and the -o argument is omitted, the output 
file will be FILE.generated.EXT

If both the -s and -o arguments are omited and we have -f FILE.EXT, the output
file will be FILE.simpedia.generated.xml
"""

from skimage.transform import hough, probabilistic_hough
from skimage.filter import canny
from skimage import data
from skimage.io import imread

import numpy as np
import matplotlib.pyplot as plt
import yaml

import pdb
import argparse

SOURCEFILE = 'data/cpdFloorPlan.pgm'
TRANSFORMPARAMS = {'threshold':10, 'line_length':5, 'line_gap':3}
PLOT = True


__file__ = 'map_generate.py'

from xml.etree import ElementTree as et


def scale_up(image):
  """
  @return an image which is scaled up by 2x
  """
  ret = np.zeros(np.array(np.shape(image))*2)
  for i in 0,1:
    for j in 0,1:
      ret[i::2,j::2] = image
  return ret

def shrink_lines(image, threshold=128, iterations=1):
  """
  Each iteration entails clearing cells that are neighbors of 4-connected
  cells that are not themselves 4-connected.
  @param threshold: assuming the original image is grayscale, convert to b/w
  with this threshold
  @param iterations: the number of times to shrink
  """
  bwdata = np.where(image<128,1,0)
  for iter in range(iterations):
    bwdata = shrink_once(bwdata)
  return np.amax(image)*bwdata

def shrink_once(bwdata):
  """
  @param b/w image
  @return a new image in which we have cleared cells that are neighbors of 
  4-connected cells that are not themselves 4-connected.
  """
  ijs = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
  #for iter in range(iterations):
  #compute 4 connected image
  bw4c = np.ones(np.array(np.shape(bwdata))-2)
  for i,j in ijs:
    bw4c *= bwdata[1+i:(-1+i) or None,1+j:(-1+j) or None]

  # compute neighbors of 4-connected
  n4c = np.ones(np.shape(bwdata))
  for i,j in ijs:
    n4c[1+i:(-1+i) or None,1+j:(-1+j) or None] *= (bw4c==0)*1
  n4c = np.where(n4c==0,1,0)
  # compute fringe of 4-connected
  n4c[1:-1,1:-1] -= bw4c
  # subtract fringe
  ret = bwdata - n4c*bwdata
  return ret

def find_lines(sourcefile=SOURCEFILE, plot=PLOT, shrink=2, threshold=128,
               scale_first=True,
               **transformparams):
  """
  This function borrows from
  http://scikits-image.org/docs/0.3/auto_examples/plot_hough_transform.html
  """
  # Line finding, using the Probabilistic Hough Transform
  params = {}
  params.update(TRANSFORMPARAMS)
  params.update(transformparams)
  print params
  image_orig = imread(sourcefile)

  # scale up
  if scale_first:
    image_orig = scale_up(image_orig)
  # thin lines
  if shrink > 0:
    # note that this returns a one-bit image
    shrunk_image = shrink_lines(image_orig,threshold=threshold,
                                iterations=shrink)
  else:
    # in this case shrunk_image may still be grayscale
    shrunk_image = image_orig
  # switch the y axis
  image = shrunk_image[::-1,:]
  #edge detection
  edges = canny(image, 2, 1, 25)
  #actual transform
  lines = probabilistic_hough(edges, **params)

  if PLOT:
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Input image')

    plt.subplot(132)
    plt.imshow(edges, cmap=plt.cm.gray)
    plt.title('Sobel edges')

    plt.subplot(133)
    plt.imshow(edges * 0)

    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

    plt.title('Lines found with PHT')
    plt.axis('image')
    plt.show()
  if scale_first:
    lines = np.array(lines)/2
  return lines


def et_from_lines(lines,sourcefile=SOURCEFILE,yamlfile=None,
                transformparams=TRANSFORMPARAMS,wall_thickness=.005,**kwargs):
  """
  @param lines [((x0,y0),(x1,y1))]* or np.array(same)
  @returns ElementTree with root = <section name="obstacles"/>
  """
  if yamlfile is None:
    # guess the name
    yamlfile = sourcefile[:-3]+'yaml'
  with open(yamlfile) as fd:
    yamlparams = yaml.load(fd)

  # create root element
  root = et.Element('section',attrib={'name':'obstacles'})
  root.append(et.Comment("""
This section was generated via python script {0}
from the image {1} using The Probabilistic Hough Transform with parameters
{2}""".format(__file__,sourcefile,transformparams)))

  #convert lines to offset (c) ,length (l), theta with proper origin and scale
  laint = np.array(lines)
  la = laint*yamlparams['resolution']
  x0,x1,y0,y1 = la[:,0,0],la[:,1,0],la[:,0,1],la[:,1,1]
  l = np.sqrt(sum(np.array([x1-x0,y1-y0])**2))
  theta = np.arctan2(y1-y0,x1-x0)
  c = (np.array([x1+x0,y1+y0])/2.).T + np.array(yamlparams['origin'][:2])

  #create Elements for each
  for i in range(len(theta)):
    root.append(et.Element('entry',{
          'key':'obstacle.{0}'.format(i),
          'value':'true {0[0]} {0[1]} 0 {1} {3} 3 0 0 {2}'.format(
            c[i],l[i],theta[i],wall_thickness)}))

  tree = et.ElementTree(root)
  return tree

def write_lines(lines,filename,**kwargs):
  """
  simply write tree in xml-format
  """
  tree = et_from_lines(lines,**kwargs)
  tree.write(filename)

def update_from_lines(lines,startfile,endfile=None,**kwargs):
  """
  replace or add an obstacles tag to an existing xml file
  """
  if endfile is None:
    endfile=startfile+'.mod'
  tree = et.parse(startfile)
  root = tree.getroot()
  if root is None:
    print "could not create an elementr tree from file %s"%startfile
    return None
  sections = root.findall('section')
  obstacles = [s for s in sections if s.attrib['name']=='obstacles'][0]
  root.remove(obstacles)
  
  newobstacles = et_from_lines(lines,**kwargs)
  root.append(newobstacles.getroot())
  tree.write(endfile)
  

######################
# Argument parsing   #
######################
parser = argparse.ArgumentParser(description=usage)
parser.add_argument('-f','--file', help='source image file', required=True)
parser.add_argument('-s','--simpedia', help=(
    'base simpedia file.  If the file exists, the script will replace the'
    '<obstacles> element.'), default=None, required=False)
parser.add_argument('-y','--yaml', help='yaml file describing the image', 
                    required=False, default=None)
parser.add_argument('-o','--output', help='output file name', required=False)
parser.add_argument('-t','--threshold',
                    help='Hough Accumulator threshold, default 10', 
                    type=int, required=False, default=10)
parser.add_argument('-l', '--line_length', 
                    help='Minimum segment length (px), default 5', 
                    type=int, required=False, default=5)
parser.add_argument('-g', '--line_gap', default=3,type=int,required=False,
                    help="minimum gap between distinct lines (px) default 3")
parser.add_argument('-w','--wall_thickness', default=.005,type=float,
                    required=False, help=('width of simpedia walls (m)'
                                          'default .005'))
parser.add_argument('-p','--plot',type=bool,required=False,default=PLOT,
                    help='display results, default %s'%(PLOT))
parser.add_argument('-S','--shrink',type=int, required=False, default=2,
                    help='number of times to reduce the fringe of 4-connected '
                    'cells')
parser.add_argument('-T','--bw_threshold', type=int, required=False,default=128,
                    help='threshold for converting to B/W from grayscale if '
                    'necessary')
parser.add_argument('-u','--scale_up',type=bool,default=True,
                    help='before shrinking, double the image scale')
args_ns = parser.parse_args()



if __name__=='__main__':
  main()

def main():
  args = vars(args_ns)
  
  source = args['file']
  sim = args['simpedia']

  if args['output'] is None:
    if sim is None:
      args['output'] = source[:source.rfind('.')+1]+'simpedia.generated.xml'
    else:
      args['output'] = sim[:sim.rfind('.')+1]+'generated'+sim[sim.rfind('.'):]

  if args['yaml'] is None:
    args['yaml'] = source[:source.rfind('.')+1]+'yaml'

  transformparams = {'threshold':args['threshold'], 
                     'line_length':args['line_length'], 
                     'line_gap':args['line_gap']}


  lines = find_lines(sourcefile=source,plot=args['plot'],shrink=args['shrink'],
                     threshold=args['bw_threshold'],
                     scale_first=args['scale_up'])
  if sim is None:
    write_lines(lines,args['output'],sourcefile=source,yamlfile=args['yaml'],
                transformparams=transformparams, 
                wall_thickness=args['wall_thickness'])
  else:
    update_from_lines(lines,sim,endfile=args['output'],
                      sourcefile=source,yamlfile=args['yaml'],
                transformparams=transformparams, 
                wall_thickness=args['wall_thickness'])
