import os,sys,random,time
import carla
import numpy as np
import cv2

#settings
IM_W,IM_H = (420,280)
time_step = 1.5
image_save_path ='_data'
seq_len = 15
number_env_vehicles = 35
if not os.path.exists(os.path.join(image_save_path)):
    os.makedirs(os.path.join(image_save_path))
#create main carla objects
client = carla.Client('localhost',2000)
client.set_timeout(5)
world = client.get_world()

blueprint_library = world.get_blueprint_library()

class Carla_session:

    def __init__(self):
        self.actors = []
        self.counter = 0
        self.n_seq = len(os.listdir(image_save_path))
        self.collision_flag = False
        self.episode_images = []
        self.track_cleanup =[]
        self.env_actors = []

    def add_vehicles(self):
        env_vehicles_bp = blueprint_library.filter('vehicle.*')
        env_vehicles_bp = [x for x in env_vehicles_bp if int(x.get_attribute('number_of_wheels')) == 4]
        env_vehicles_bp = [x for x in env_vehicles_bp if not x.id.endswith('isetta')]
        env_vehicles_bp = [x for x in env_vehicles_bp if not x.id.endswith('carlacola')] 
        spawn_points = world.get_map().get_spawn_points()      
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        self.env_actors = []
        for n, transform in enumerate(spawn_points):
            if n >= number_env_vehicles:
                break
            env_vehicle_bp = random.choice(env_vehicles_bp)
            if env_vehicle_bp.has_attribute('color'):
                env_vehicle_bp.set_attribute('color', random.choice(env_vehicle_bp.get_attribute('color').recommended_values))
            env_vehicle_bp.set_attribute('role_name', 'autopilot')
            env_vehicle = world.spawn_actor(env_vehicle_bp,transform)
            env_vehicle.set_autopilot(True)
            self.env_actors.append(env_vehicle)

    def add_actors(self):

        start_point = random.choice(world.get_map().get_spawn_points())

        #set vehicle
        vehicle_bp = blueprint_library.find('vehicle.ford.mustang')
        self.vehicle = world.spawn_actor(vehicle_bp,start_point)
        #self.vehicle.set_autopilot(True)

        #get and set sensors
        collision_sensor_bp = blueprint_library.find('sensor.other.collision')
        lane_invasion_sensor_bp = blueprint_library.find('sensor.other.lane_invasion')
        camera_sensor_bp = blueprint_library.find('sensor.camera.rgb')
        camera_sensor_bp.set_attribute('image_size_x',str(IM_W))
        camera_sensor_bp.set_attribute('image_size_y',str(IM_H))
        #camera_sensor_bp.set_attribute('sensor_tick',str(time_step))
        camera_sensor_bp.set_attribute('fov',str(100))

        sensor_location = carla.Transform(carla.Location(x=4,y=0,z=2.5))
        self.camera = world.spawn_actor(camera_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.collision_sensor = world.spawn_actor(collision_sensor_bp, sensor_location, attach_to = self.vehicle)
        #self.lane_invasion_sensor = world.spawn_actor(lane_invasion_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.actors.extend([self.vehicle,self.camera,self.collision_sensor])
        self.camera.listen(lambda image: self.add_image(image))
        self.collision_sensor.listen(lambda collision: self.end_seq(collision,'collision'))  

        #self.lane_invasion_sensor.listen(lambda lane_inv: self.end_seq(lane_inv,'crossed lane'))

    def start_new_seq(self):
        
        self.add_actors()
        self.collision_flag = False
        print('starting new seq')
        self.counter = 0
        self.n_seq+=1
        self.track_cleanup.append(self.n_seq)
        if not os.path.exists(os.path.join(image_save_path,str(self.n_seq))):
            os.makedirs(os.path.join(image_save_path,str(self.n_seq)))

    def add_image(self,image):
        self.counter += 1
        img = np.reshape(image.raw_data,(IM_H,IM_W,4))
        img = img[:,:,:3][:]
        #self.episode_images.append(img)
        
        cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.png'.format(self.counter)),img)
        '''if self.counter%15 == 0:
            self.n_seq += 1
            self.counter = 0
            if not os.path.exists(os.path.join(image_save_path,str(self.n_seq))):
                os.makedirs(os.path.join(image_save_path,str(self.n_seq)))'''

        #cv2.imshow("live",img)
        #cv2.waitKey(1)

    def delete_images(self):
        imagestodelete = self.counter-seq_len
        for i in range(imagestodelete):
            os.remove(os.path.join(image_save_path,str(self.n_seq),'{}.png'.format(i+1)))

    def save_images(self):
        #print(os.path.join(image_save_path,str(self.n_seq),'{}.png'.format(self.counter)))
        for ind,img in enumerate(self.episode_images[-seq_len:]):
            cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.png'.format(ind)),img)
    
    def end_seq(self,cause_obj,cause):
        self.destroy_actors()
        self.collision_flag =True
        print("collision happened")
        self.delete_images()

    def destroy_actors(self):
        for actor in self.actors:
            print(actor)
            actor.destroy()

        #self.save_images()
        self.actors = []
        #self.episode_images =[]
    
    def get_directions(self):
        thr = random.choice([0.8,0.7,0.6])
        steer = random.choice([-0.3,0.0,0.0,0.0,0.3,0.1,-0.1])
        return carla.VehicleControl(thr,steer)  
       
    def drive_around(self,episodes):
        self.add_vehicles()
        
        for i in range(episodes):
            try:
                self.start_new_seq()
                for j in range(200):
                    self.vehicle.apply_control(self.get_directions())
                    time.sleep(1)
                    if self.collision_flag == True:
                        break
                    
            except Exception as e:
                print(e)
                continue

        '''for i in self.env_actors:
            i.destory()
        self.env_actors =[]'''


c = Carla_session()
c.drive_around(10)