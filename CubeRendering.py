import bpy
import math


cam = bpy.data.objects['Camera']
origin = bpy.data.objects['Empty']
cube = bpy.data.objects['Cube']
C = bpy.context.object

step_count = 32
#Point light source
C.data.type = 'POINT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_pointLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )
    
#Sun light source    
C.data.type = 'SUN'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_SunLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Spot light source    
C.data.type = 'SPOT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_SpotLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'HEMI'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_HemiLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'AREA'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_AreaLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

print ('End part normal')

bpy.context.scene.world.light_settings.use_ambient_occlusion = True

#Point light source
C.data.type = 'POINT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_AO_pointLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )
    
#Sun light source    
C.data.type = 'SUN'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_AO_SunLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Spot light source    
C.data.type = 'SPOT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_AO_SpotLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'HEMI'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_AO_HemiLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'AREA'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_AO_AreaLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

bpy.context.scene.world.light_settings.use_ambient_occlusion = False
print ('End AO')

bpy.context.scene.world.light_settings.use_environment_light = True

#Point light source
C.data.type = 'POINT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_EL_pointLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )
    
#Sun light source    
C.data.type = 'SUN'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_EL_SunLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Spot light source    
C.data.type = 'SPOT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_EL_SpotLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'HEMI'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_EL_HemiLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'AREA'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_EL_AreaLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )
    
bpy.context.scene.world.light_settings.use_environment_light = False

print ('End EL')

bpy.context.scene.world.light_settings.use_indirect_light = True
#Point light source
C.data.type = 'POINT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_IL_pointLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )
    
#Sun light source    
C.data.type = 'SUN'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_IL_SunLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Spot light source    
C.data.type = 'SPOT'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_IL_SpotLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'HEMI'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_IL_HemiLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )

#Sun light source    
C.data.type = 'AREA'

for step in range(0, step_count):
    cube.rotation_euler[2] = math.radians(step * (360.0 / step_count))

    bpy.data.scenes["Scene"].render.filepath = '/Users/karri/Desktop/6cp/rendered/render_IL_AreaLight_%d.jpg' % step
    bpy.ops.render.render( write_still=True )
    
bpy.context.scene.world.light_settings.use_indirect_light = False

print ('End IL')


    


