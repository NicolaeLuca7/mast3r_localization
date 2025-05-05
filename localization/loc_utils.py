import numpy as np

class Position():
    def __init__(self,relative,absolute,heading):
        self.relative = relative
        self.absolute = absolute
        self.heading = heading

class Matches():
    def __init__(self,satellite,ground,total,h_matrix):
        self.satellite = satellite
        self.ground = ground
        self.total = total
        self.h_matrix = h_matrix

class LocOutput():
    def __init__(self,position:Position,matches:Matches):
        self.position = position
        self.matches = matches 


class Patches():
    def __init__(self,images,centers):
        self.images = images
        self.centers = centers

class CameraSpecs():
    def __init__(self,hfov,vfov,width,height):
        self.hfov = hfov
        self.vfov = vfov
        self.width = width
        self.height = height
        self.fx =  width / (2*np.tan(np.deg2rad(hfov)/2))
        self.fy = height / (2*np.tan(np.deg2rad(vfov)/2))
        self.cx = width  / 2
        self.cy = height / 2

class DroneState():
    def __init__(self,alt,pitch,roll):
        self.alt = alt
        self.pitch = pitch
        self.roll = roll

class Map_Info():
    def __init__(self,
                 map_gsd,
                 patch_size,
                 meters_lat_degree,meters_lon_degree,
                 map_pos,
                 alt_offset,
                 map_img,
                 ground_imgs,
                 camera_specs, 
                 csv_path = None,
                 map_tif = None,
                 satellite_imgs = None):
        self.map_gsd = map_gsd
        self.patch_size = patch_size #meters
        self.meters_lat_degree = meters_lat_degree
        self.meters_lon_degree = meters_lon_degree
        self.map_pos = map_pos
        self.alt_offset = alt_offset
        self.map_img = map_img
        self.map_tif = map_tif
        self.ground_imgs = ground_imgs
        self.satellite_imgs = satellite_imgs
        self.camera_specs = camera_specs
        self.csv_path = csv_path



