import bpy, sys, os
from math import *

cam = bpy.data.objects['Camera']
origin = bpy.data.objects['Empty']
cube = bpy.data.objects['Cube']

step_count = 32

for step in range(0, step_count):
    orgin.rotation_euler[2] = radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_shot_cam_%d.jpg' % step
    bpy.ops.render.render( write_still=True )