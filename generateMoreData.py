
import bpy
from bpy.props import *
from math import *

print ("working")

context           = bpy.context
scene             = bpy.context.scene
frame             = scene.frame_current
obj               = bpy.data.objects["Camera"]
camspheremesh     = bpy.data.meshes["CameraBoss"] 
camsphereobj      = bpy.data.objects["CameraBoss"]
vertamount        = len(camspheremesh.vertices)
size              = camsphereobj.scale
scene.frame_start = 1
scene.frame_end   = vertamount


# Store properties in the active scene 
def initSceneProperties(scn):
	bpy.types.Scene.levels = IntProperty(name="Levels", description="amount of vertical levels", default=7, min=1, max=15)	
	bpy.types.Scene.degrees = IntProperty(name="Degrees", description="horizontal degrees for camera rotation", default=10, min=1, max=45)	
	bpy.types.Scene.size = IntProperty(name="Size", description="size of the CameraBoss object", default=20, min=1, max=200)	
	bpy.types.Scene.capbuffer = FloatProperty(name="CapBuffer", description="space between the cap and the first camera level", default=2.0, min=0.5, max=10.0)

initSceneProperties(scene)
	

# Set keyframes for CameraBoss
def setKeyFrames():
	vertamount = len(camspheremesh.vertices)
	
	# remove existing keyframes
	for i in range( 1, 10000 ):
		# Blender 2.5x:
		#obj.keyframe_delete(data_path="location", frame=i) 
		# Blender 2.6x:
		obj.animation_data_clear()
		scene.frame_current = 1
	
	# add new keyframes	
	for i in range( 1, vertamount+1 ):
		vert = camspheremesh.vertices[i-1]
		obj.location = (vert.co[0]*size[0], vert.co[1]*size[1], vert.co[2]*size[2]) 
		obj.keyframe_insert(data_path="location", frame=i)
		#print ("########## Keyframe added:",i,"##########")
		

# make one level of the sphere
def makeLevel(steps, z, width):

	for i in range(90, steps+90):

		winkel = i*pi/steps*2
		x = -(sin(winkel)*width)
		y = cos(winkel)*width
		
		# Add one vertice and set coordinates
		camspheremesh.vertices.add(1)
		camspheremesh.vertices[len(camspheremesh.vertices)-1].co = (x,y,z)
		

# make the CameraBoss
def makeCameraBoss(levels,degrees,size,capbuffer):
	global me
	total = 360/degrees*levels
	
	# make the sphere
	try:
		cameraboss = camsphereobj
		me = camspheremesh
		wasthere = 'true'
		#print ("Was there: true")
		
	except:
		cameraboss = Blender.Object.New('Mesh')
		cameraboss.setName('CameraBoss') 
		me = NMesh.New('CameraBoss')
		wasthere = 'false'
		#print ("Was there: false")		

	if wasthere == 'true':
		# select all vertices
		for f in camsphereobj.data.vertices:
			f.select = True
			
		if bpy.context.mode == 'OBJECT':
			bpy.ops.object.editmode_toggle()

		# delete selected vertices			
		bpy.ops.mesh.delete(type='VERT')
		bpy.ops.object.editmode_toggle()
		me.update() 

	# always do this
	for m in range(0, levels):
		steps = int(360/degrees)
		if levels > 1:
			ystepper = ((size-capbuffer)/(levels-1))
			z = (size/2)-(capbuffer/2)-(m*ystepper)
		else:
			z=0	
		width = sqrt(((size/2)*(size/2))-(z*z))
		makeLevel (steps, z, width)
		me.update()

	if wasthere != 'true':
		cameraboss.link(me)
		scene.link (cameraboss)


# GUI
class ToolsPanel(bpy.types.Panel):

	bl_label = "GenerateMoreData Pannel"
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"


	def draw(self, context):
		layout = self.layout
		layout.prop(scene, 'levels')		
		layout.prop(scene, 'degrees')
		layout.prop(scene, 'size')
		layout.prop(scene, 'capbuffer')
		row = layout.row()
		row.alignment = 'RIGHT'
		row.operator("my.button", text="       Update Camera")

class OBJECT_OT_Button(bpy.types.Operator):
	bl_idname = "my.button"
	bl_label = "Button"
	def execute(self, context):
		makeCameraBoss(scene['levels'],scene['degrees'],scene['size'],scene['capbuffer'])
		setKeyFrames()
		scene.frame_end = len(camspheremesh.vertices)
		return{'FINISHED'}    

# Register
bpy.utils.register_module(__name__)