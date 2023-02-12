import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader 
import pyrr
from PIL import Image
import math

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_END = 0

def initialize_glfw():

    glfw.init()
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_PROFILE,
        GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
    )

    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT,
        GLFW_CONSTANTS.GLFW_TRUE
    )
    glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, GL_FALSE)

    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "My Game", None, None)
    glfw.make_context_current(window)
    glfw.set_input_mode(
        window,
        GLFW_CONSTANTS.GLFW_CURSOR,
        GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN
    )
    return window


class SimpleComponent:
    def __init__(self, positions, eulers, scales):
        self.position = np.array(positions, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.scales = np.array(scales, dtype=np.float32)
        pass

class Light:

    def __init__(self, position, color, strength):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

        pass

class Player:
    
    def __init__(self, position):
        self.position = np.array(position, dtype= np.float32)
        self.theta = 0
        self.phi = 0
        self.update_vectors()
        pass

    def update_vectors(self):
        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.phi))

            ]
        )
        globalUp = np.array([0,0,1], dtype=np.float32)
        self.right = np.cross(self.forwards, globalUp)
        self.up = np.cross(self.right, self.forwards)
        pass

class Scene:

    def __init__(self):

        self.components = [
            SimpleComponent(
                positions=
                [6,0,0],
                eulers=[-90,0,90],
                scales=[0.5,0.5,0.5]
            )
        ]
        self.lights = [
            Light(
                position= [
                    np.random.uniform(low=3.0, high=9.0),
                    np.random.uniform(low=-2.0, high=2.0),
                    np.random.uniform(low=2.0, high=4.0),
                ],
                color=[
                    np.random.uniform(low=0.0, high=0.0),
                    np.random.uniform(low=0.5, high=1.0),
                    np.random.uniform(low=0.5, high=1.0),
                ],
                strength=3
            )
            for i in range(8)
        ]
        self.player = Player(position=[0,0,2])
        pass
    
    def update(self, rate):
        for component in self.components:
            component.eulers[2] += 0.25 * rate
            if component.eulers[2] > 360:
                component.eulers[2] -= 360
        pass

    def move_player(self, dPos):
        dPos = np.array(dPos, dtype=np.float32)
        self.player.position += dPos
        pass
    
    def spin_player(self, dTheta, dPhi):
        self.player.theta += dTheta
        if self.player.theta > 360:
            self.player.theta -= 360
        elif self.player.theta < 0:
            self.player.theta += 360
        
        self.player.phi = min(89, max(-89, self.player.phi + dPhi))
        self.player.update_vectors()
        pass

class App:
    def __init__(self, window):

        self.window = window
        self.renderer = GraphicsEngine()
        self.scene = Scene()

        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.walk_offset_lookup = {
            1: 0,
            2: 90,
            3: 45,
            4: 180,
            6: 135,
            7: 90,
            8: 270,
            9: 315,
            11: 0,
            12: 225,
            13: 270,
            14: 180
        }

        self.mainLoop()
        pass


    def mainLoop(self):

        running = True
        c_time = glfw.get_time()
        last_time = c_time
        while (running):
            #check events
            if glfw.window_should_close(self.window) \
                or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:

                running = False
            c_time = glfw.get_time()
            deltaTime = c_time - last_time
            last_time = c_time
            self.handleKeys(deltaTime)
            self.handleMouse()
            
            glfw.poll_events()

            self.scene.update(deltaTime * 100)

            self.renderer.render(self.scene)

            #timing
            self.calculateFramerate()
        self.quit()
    
    def handleKeys(self, deltaTime):
        combo = 0
        directionModifier = 0

        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_W) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 1
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_A) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 2
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_S) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 4
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_D) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 8
        
        if combo in self.walk_offset_lookup:
            directionModifier = self.walk_offset_lookup[combo]
            dPos = [
                deltaTime * np.cos(np.deg2rad(self.scene.player.theta + directionModifier)),
                deltaTime * np.sin(np.deg2rad(self.scene.player.theta + directionModifier)),
                0
            ]
            self.scene.move_player(dPos)
        pass

    def handleMouse(self):

        (x,y) = glfw.get_cursor_pos(self.window)
        rate = self.frameTime / 16.7
        theta_increment = rate * ((SCREEN_WIDTH/2) - x)
        phi_increment = rate * ((SCREEN_HEIGHT/2) - y)
        self.scene.spin_player(theta_increment, phi_increment)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def calculateFramerate(self):
        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            framerate = max(1, int(self.numFrames / delta))
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0/max(1,framerate))
        self.numFrames += 1

        pass

    def quit(self):
        self.renderer.quit()

class GraphicsEngine:

    def __init__(self):

        # self.wood_texture = Material("gfx/wood.png")
        # self.cube_mesh = Mesh("model/nanosuit.obj")
        self.renderObj = RenderObj("model/nanosuit.obj")

        self.light_billboard = BillBorad(w = 0.2, h = 0.2)
        #initialize opengl
        glClearColor(0.1, 0.2, 0.2, 1)
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        self.LitShader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        self.textureLitPass = RenderPassTextureLit3D(self.LitShader)
        self.shader = self.createShader("shaders/vertex_light.txt", "shaders/fragment_light.txt")
        self.texturePass = RenderPassTexture3D(self.shader)
        pass
    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()
        
        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        return shader
    def render(self, scene):
        
        #refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        

        self.textureLitPass.render(scene, self)
        self.texturePass.render(scene, self)
        glFlush()
        pass
    def quit(self):
        # self.cube_mesh.destroy()
        # self.wood_texture.destroy()
        self.renderObj.destroy()
        glDeleteProgram(self.shader)
        glDeleteProgram(self.LitShader)

class RenderPassTextureLit3D:
    
    def __init__(self, shader):
        self.shader = shader
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=640/480, near=0.1, far=50, dtype=np.float32
        )
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection_transform)
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.lightLocation = {
            "position": [glGetUniformLocation(self.shader, f"Lights[{i}].position")
            for i in range(8)
            ],
            "color": [glGetUniformLocation(self.shader, f"Lights[{i}].color")
            for i in range(8)
            ],
            "strength": [glGetUniformLocation(self.shader, f"Lights[{i}].strength")
            for i in range(8)
            ]
        }
        self.cameraPosLoc = glGetUniformLocation(self.shader, "cameraPosition")
        pass
    def render(self, scene, engine):
        glUseProgram(self.shader)

        view_transform = pyrr.matrix44.create_look_at(
            eye=scene.player.position,
            target=scene.player.position + scene.player.forwards,
            up=scene.player.up, dtype=np.float32
        )
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)
        glUniform3fv(self.cameraPosLoc, 1, scene.player.position)

        for i, light in enumerate(scene.lights):
            glUniform3fv(self.lightLocation["position"][i], 1, light.position)
            glUniform3fv(self.lightLocation["color"][i], 1, light.color)
            glUniform1f(self.lightLocation["strength"][i], light.strength)
            pass

        for component in scene.components:
            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(m1=model_transform, m2=pyrr.matrix44.create_from_eulers(eulers=np.radians(component.eulers), dtype=np.float32))
            model_transform = pyrr.matrix44.multiply(m1=model_transform, m2=pyrr.matrix44.create_from_scale(scale=np.array(component.scales), dtype=np.float32))
            model_transform = pyrr.matrix44.multiply(m1=model_transform, m2=pyrr.matrix44.create_from_translation(vec=np.array(component.position), dtype=np.float32))
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            engine.renderObj.render()
            pass
        pass

    def destroy(self):
        pass
    pass

class RenderPassTexture3D:
    def __init__(self, shader):
        self.shader = shader
        glUseProgram(self.shader)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=640/480,
            near=0.1, far= 50, dtype=np.float32
        )
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection_transform)

        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.lightColorLocation = glGetUniformLocation(self.shader, "lightColor")

        pass
    def render(self, scene, engine):
        glUseProgram(self.shader)

        view_transform = pyrr.matrix44.create_look_at(
            eye=scene.player.position,
            target=scene.player.position + scene.player.forwards,
            up=scene.player.up, dtype=np.float32
        )

        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)

        for i, light in enumerate(scene.lights):
            glUniform3fv(self.lightColorLocation, 1, light.color)

            directionFromPlayer = light.position - scene.player.position
            angle1 = np.arctan2(-directionFromPlayer[1],directionFromPlayer[0])
            dist2d = math.sqrt(directionFromPlayer[0] ** 2 + directionFromPlayer[1] ** 2)
            angle2 = np.arctan2(directionFromPlayer[2],dist2d)

            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_y_rotation(theta=angle2, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_z_rotation(theta=angle1, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_translation(light.position,dtype=np.float32)
            )
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            glBindVertexArray(engine.light_billboard.vao)
            glDrawArrays(GL_TRIANGLES, 0, engine.light_billboard.vertex_count)
            pass
    pass
class CubeMesh:
    def __init__(self):
          # x, y, z, s, t
        self.vertices = (
                -0.5, -0.5, -0.5, 0, 0,
                 0.5, -0.5, -0.5, 1, 0,
                 0.5,  0.5, -0.5, 1, 1,

                 0.5,  0.5, -0.5, 1, 1,
                -0.5,  0.5, -0.5, 0, 1,
                -0.5, -0.5, -0.5, 0, 0,

                -0.5, -0.5,  0.5, 0, 0,
                 0.5, -0.5,  0.5, 1, 0,
                 0.5,  0.5,  0.5, 1, 1,

                 0.5,  0.5,  0.5, 1, 1,
                -0.5,  0.5,  0.5, 0, 1,
                -0.5, -0.5,  0.5, 0, 0,

                -0.5,  0.5,  0.5, 1, 0,
                -0.5,  0.5, -0.5, 1, 1,
                -0.5, -0.5, -0.5, 0, 1,

                -0.5, -0.5, -0.5, 0, 1,
                -0.5, -0.5,  0.5, 0, 0,
                -0.5,  0.5,  0.5, 1, 0,

                 0.5,  0.5,  0.5, 1, 0,
                 0.5,  0.5, -0.5, 1, 1,
                 0.5, -0.5, -0.5, 0, 1,

                 0.5, -0.5, -0.5, 0, 1,
                 0.5, -0.5,  0.5, 0, 0,
                 0.5,  0.5,  0.5, 1, 0,

                -0.5, -0.5, -0.5, 0, 1,
                 0.5, -0.5, -0.5, 1, 1,
                 0.5, -0.5,  0.5, 1, 0,

                 0.5, -0.5,  0.5, 1, 0,
                -0.5, -0.5,  0.5, 0, 0,
                -0.5, -0.5, -0.5, 0, 1,

                -0.5,  0.5, -0.5, 0, 1,
                 0.5,  0.5, -0.5, 1, 1,
                 0.5,  0.5,  0.5, 1, 0,

                 0.5,  0.5,  0.5, 1, 0,
                -0.5,  0.5,  0.5, 0, 0,
                -0.5,  0.5, -0.5, 0, 1
            )
        self.vertex_count = len(self.vertices)//5
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        pass
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo, ))
        pass

class Triangle:

    def __init__(self):

        #x, y, z, r, g, b
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
             0.0,  0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0,
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = 3

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao, ))
        glDeleteBuffers(1, (self.vbo, ))
        pass

class RenderObj:
    def __init__(self, filepath):
        self.meshs = []
        self.materials = {}

        vertices_group, materials_index = self.loadMesh(filepath)
        self.materials_index = materials_index
        for i in range(len(vertices_group)):
            self.meshs.append(VMesh(vertices_group[i]))

            if materials_index[i] not in self.materials.keys():
                self.materials[materials_index[i]] = Material( "gfx/"+ materials_index[i].lower() + "_dif.png")
                pass
            # self.materials.append(Material())
            # self.materials.append(Material())
        pass

    def render(self):
        for i in range(len(self.meshs)):
            self.meshs[i].bind()
            if self.materials_index[i] in self.materials:
                self.materials[self.materials_index[i]].use()
            glDrawArrays(GL_TRIANGLES, 0, self.meshs[i].vertex_count)
        pass

    def destroy(self):
        for i in self.meshs:
            i.destroy()
        
        for i in self.materials:
            self.materials[i].destroy()

        pass

    def loadMesh(self, filename):

        #raw, unassembled data
        v = []
        vt = []
        vn = []
        
        #final, assembled and packed result
        vertices = []
        vertices_group = []
        materials_index = []
        current_name = ""

        #open the obj file and read the data
        with open(filename,'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag == "usemtl":
                    if current_name != "":
                        vertices_group.append(vertices)
                        materials_index.append(current_name)
                    vertices = []
                    line = line.replace("usemtl ","")
                    current_name = line.replace("\n", "")
                if flag=="v":
                    #vertex
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag=="vt":
                    #texture coordinate
                    line = line.replace("vt ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag=="vn":
                    #normal
                    line = line.replace("vn ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag=="f":
                    #face, three or more vertices in v/vt/vn form
                    line = line.replace("f ","")
                    line = line.replace("\n","")
                    #get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        #break out into [v,vt,vn],
                        #correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        #last        
        if current_name != "":
            vertices_group.append(vertices)
            materials_index.append(current_name)

        return vertices_group, materials_index
class Material:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(filepath, mode = "r") as image:
            image_width,image_height = image.size
            image = image.convert("RGBA")
            img_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        pass
    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        pass
    def destroy(self):
        glDeleteTextures(1, (self.texture,))

class BillBorad:

    def __init__(self, w, h):
        #x,y,z, s,t, normal
        self.vertices = (
            0, -w/2,  h/2, 0, 0, -1, 0, 0,
            0, -w/2, -h/2, 0, 1, -1, 0, 0,
            0,  w/2, -h/2, 1, 1, -1, 0, 0,

            0, -w/2,  h/2, 0, 0, -1, 0, 0,
            0,  w/2, -h/2, 1, 1, -1, 0, 0,
            0,  w/2,  h/2, 1, 0, -1, 0, 0
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = 6

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        pass

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao, ))
        glDeleteBuffers(1, (self.vbo, ))
        pass

class VMesh:
    def __init__(self, vertices):
        self.vertices = vertices
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

        self.unbind()
    
        pass
    def bind(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo) 
        pass

    def unbind(self):
        
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        pass
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))
        pass
class Mesh:
    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    def loadMesh(self, filename):

        #raw, unassembled data
        v = []
        vt = []
        vn = []
        
        #final, assembled and packed result
        vertices = []

        #open the obj file and read the data
        with open(filename,'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag=="v":
                    #vertex
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag=="vt":
                    #texture coordinate
                    line = line.replace("vt ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag=="vn":
                    #normal
                    line = line.replace("vn ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag=="f":
                    #face, three or more vertices in v/vt/vn form
                    line = line.replace("f ","")
                    line = line.replace("\n","")
                    #get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        #break out into [v,vt,vn],
                        #correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)

                line = f.readline()
        return vertices
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

if __name__ == "__main__":
    window = initialize_glfw()
    myApp = App(window)
    