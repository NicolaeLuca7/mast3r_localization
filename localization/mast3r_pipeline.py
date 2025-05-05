import torchvision.transforms.v2 as transforms
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.image import preprocess_images
import cv2 as cv2
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image
from PIL.ExifTags import TAGS 
import time
from sklearn import datasets, linear_model
import copy
import seaborn as sns
import pandas as pd
import torch
from loc_utils import Position, Matches, LocOutput, Patches, Map_Info, CameraSpecs, DroneState


class Mast3rPipeline():
    def __init__(self,map_info,device):
        self.map_info = map_info
        
        self.device = device
        self.model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        self.model = AsymmetricMASt3R.from_pretrained(self.model_name).to(device)
        self.model.eval()
        self.model = self.model.to(torch.float16)
        
        self.map_gsd = map_info.map_gsd
        self.patch_size = map_info.patch_size 
        self.meters_lat_degree = map_info.meters_lat_degree
        self.meters_lon_degree = map_info.meters_lon_degree
        self.map_pos = map_info.map_pos

    def load_paths(self,file_paths):
        input_data=[]
        for path in file_paths:
            cimg=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
            input_data.append([cimg,cimg.shape[::-1][1:]])
        return input_data

    def load_processed_images(self,input_data,process_size):
        images = []
        for data in input_data:
            images.append(Image.fromarray(data[0]))
        images, res = preprocess_images(images, size=process_size)
        images[0]['img'] = images[0]['img'].to(torch.float16)
        images[1]['img'] = images[1]['img'].to(torch.float16)
        return inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False), res


    def predict_features(self,output,max_matches):
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=self.device, dist='dot', block_size=2**13)

        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        n_viz = max_matches
        num_matches = matches_im0.shape[0]
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)

        return [matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]], len(matches_im0)

    def resize_matches(self,matches,old_size,new_size):
        scale_factors = np.array(new_size) / np.array(old_size)
        return np.array(matches) * scale_factors

    def read_img(self,path):
        return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

    def get_h_matrix(self,img1, img2,method = cv2.RANSAC, threshold = 500.0):
        H, mask = cv2.findHomography(img1,img2, method,threshold)
        return H

    def transform_point(self,hmat,point):
        x, y, scale = hmat @ np.array([point[0],point[1],1])
        return np.array([x/scale,y/scale])

    def load_satellite_transform(self,path):
        with rasterio.open(path) as dataset:
            return {'dataset': dataset, 'transform': dataset.transform}

    def localize(self,
                satellite_img : np.array, 
                ground_img : np.array,
                camera_specs : CameraSpecs, 
                drone_state : DroneState,
                points : int = 10,
                process_size : int = 512,
                satellite_data = None,
                center_map_loc = None,) -> LocOutput:
        input_data = [[satellite_img,satellite_img.shape[::-1][1:]],
                    [ground_img,ground_img.shape[::-1][1:]]]
        new_imgs, new_res = self.load_processed_images(input_data,process_size)

        matches, total_matches = self.predict_features(new_imgs,points)

        for id in range(len(matches)):
            matches[id] = self.resize_matches(matches[id],new_res[id],input_data[id][1])

        query_matches = np.copy(matches[1])
        query_data = [np.copy(input_data[1][0]),input_data[1][1]]
        patch_matches = np.copy(matches[0])
        patch_data = [np.copy(input_data[0][0]),input_data[0][1]]

        l, r = np.floor(query_matches[:,0].min()).astype(np.int32), np.ceil(query_matches[:,0].max()).astype(np.int32) 
        u, d = np.floor(query_matches[:,1].min()).astype(np.int32), np.ceil(query_matches[:,1].max()).astype(np.int32)

        cropped_query_data = [
                            query_data[0][u:d,l:r],
                            [r-l+1,d-u+1]
                            ]
        cropped_query_matches = np.copy(query_matches)
        cropped_query_matches[:,0] -= query_matches[:,0].min()
        cropped_query_matches[:,1] -= query_matches[:,1].min()

        H = self.get_h_matrix(query_matches,patch_matches,method=cv2.RANSAC, threshold=500)
        

        center_pixel_loc = self.transform_point(H,
                            np.array([query_data[1][0]/2,query_data[1][1]/2])
                            ) # query pixel's corespondence on the map
        reference_point = self.transform_point(H,np.array([query_data[1][0]/2,query_data[1][1]/2-10]))
        heading = np.arctan2(-(reference_point[1]-center_pixel_loc[1]),reference_point[0]-center_pixel_loc[0])

        patch_w, pathc_h = patch_data[1]
        absolute_loc = self.get_global_pos(
                center_pixel_loc[0],center_pixel_loc[1],
                patch_w,pathc_h,
                satellite_data=satellite_data,
                center_loc=center_map_loc
                ) # gps coordinate of the pixel
        
        # absolute_loc = translate_to_drone_pose(absolute_loc[0], absolute_loc[1],
        #                                        center_map_loc[0], center_map_loc[1],
        #                                        meters_lon_degree, meters_lat_degree,
        #                                        query_data[1][0]/2,query_data[1][1]/2,
        #                                        camera_specs.fx, camera_specs.fy,
        #                                        camera_specs.cx, camera_specs.cy,
        #                                        np.deg2rad(heading), drone_state.pitch, drone_state.roll,
        #                                        drone_state.alt,
        #                                     )

        absolute_loc = self.get_corrected_pose(absolute_loc[0],absolute_loc[1],
                                               heading,drone_state.alt,
                                               drone_state.pitch)
        
        relative_loc = self.absolute_to_relative(patch_w,pathc_h,absolute_loc,center_map_loc)

        

        output = LocOutput(
                    position = Position(
                        relative = relative_loc,
                        absolute = absolute_loc,
                        heading = heading
                    ),
                    matches = Matches(
                        satellite = np.array(patch_matches),
                        ground = np.array(query_matches),
                        total = total_matches,
                        h_matrix = np.array(H)
                    )
                )
        return output

    def get_global_pos(self,x,y,w,h,satellite_data = None, center_loc = None):
        if satellite_data == None:
            lon = center_loc[0] + (x-w/2.0) * self.map_gsd / self.meters_lon_degree
            lat = center_loc[1] + (h/2.0-y) * self.map_gsd / self.meters_lat_degree
        else:
            x, y = satellite_data['transform'] * (x, y)
            if satellite_data['dataset'].crs != 'EPSG:4326':
                transformer = Transformer.from_crs(satellite_data['dataset'].crs, 'EPSG:4326', always_xy=True)
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y

        return np.array([lon,lat])

    def display_position(self,loc_output:LocOutput,satellite_img,ground_img,colors):
        r = 200
        loc = loc_output.position.relative
        heading = loc_output.position.heading
        st_point = np.array([loc[0],loc[1]], dtype=np.int64)
        ds_point = np.array([loc[0] + r*np.cos(heading),loc[1] - r*np.sin(heading)], dtype=np.int64)
        ground_img = np.copy(ground_img)
        output_img = np.copy(satellite_img)
        cv2.arrowedLine(output_img, st_point, ds_point, (0, 0, 255), (int)(np.min(output_img.shape[:2])*0.01), tipLength=0.5)
        
        id=0
        for p in loc_output.matches.satellite:
            output_img = cv2.circle(output_img,center=np.array(p,dtype=np.int64), radius=(int)(np.min(output_img.shape[:2])*0.01),color=colors[id],thickness=-1)
            id+=1
            if id == len(colors):
                break
        
        id=0
        for p in loc_output.matches.ground:
            ground_img = cv2.circle(ground_img,center=np.array(p,dtype=np.int64), radius=(int)(np.min(ground_img.shape[:2])*0.01),color=colors[id],thickness=-1)
            id+=1
            if id == len(colors):
                break
            
        rows, cols = (1,2)
        fig, axes = plt.subplots(rows, cols, figsize=(10,10))
        for ax, img in zip(axes.flat, [ground_img,output_img]):
            img_rgb = img
            ax.imshow(img_rgb)
            ax.axis('off') 
        plt.tight_layout()
        plt.show()

    def get_corrected_pose(self,plon,plat,heading,alt,pitch,fov=None):
        dist = alt / np.tan(-pitch)
        flon = plon - np.cos(heading) * dist / self.meters_lon_degree 
        flat = plat - np.sin(heading) * dist / self.meters_lat_degree
        return flon, flat

    def get_matching_error(self,loc_output:LocOutput,width,height):
        s_matches = loc_output.matches.satellite
        g_matches = loc_output.matches.ground

        errorx = 0
        errory = 0
        for i in range(len(g_matches)):
            py = np.dot(loc_output.matches.h_matrix ,np.array([g_matches[i][0],g_matches[i][1],1]))
            x, y = py[0]/py[2], py[1]/py[2]
            errorx += (x-s_matches[i][0])**2
            errory += (y - s_matches[i][1])**2
        error = np.sqrt(errorx/len(g_matches)) +  np.sqrt(errory/len(g_matches))

        return error

    def output_duration(self,start_time):
        stop_time=time.time()
        duration =stop_time - start_time
        hours = duration // 3600
        minutes = (duration - (hours * 3600)) // 60
        seconds = duration - ((hours * 3600) + (minutes * 60))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print (msg, flush=True) 

    def get_exif_data(self,image_path):
        img = Image.open(image_path)
        exif_data = img._getexif() 

        if exif_data is None:
            return "No EXIF data found."

        readable_exif_data = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag) 
            readable_exif_data[tag_name] = value
        return readable_exif_data

    def filter_gps_data(self,image_path):
        exif_data = self.get_exif_data(image_path)

        return np.array([exif_data['GPSInfo'][4][0]*1.0 + exif_data['GPSInfo'][4][1]*1.0 / 60 + exif_data['GPSInfo'][4][2]*1.0 / 3600,
                            exif_data['GPSInfo'][2][0]*1.0 + exif_data['GPSInfo'][2][1]*1.0 / 60 + exif_data['GPSInfo'][2][2]*1.0 / 3600,
                            exif_data['GPSInfo'][6]])

    def rotate_img(self,img,degrees):
        img = np.array(transforms.ToPILImage()(transforms.RandomRotation(degrees=(degrees,degrees))(
            transforms.ToTensor()(img))))
        return img

    def img_to_grayrgb(self,cimg):
        return cv2.cvtColor(cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)

    def get_patch(self,map_img, center,width,height):
        img_size = np.ceil(self.patch_size / self.map_gsd)
        img_size = (int)(img_size-img_size%2)
        sy, ey = (int)(center[1]) - img_size // 2, (int)(center[1]) + img_size // 2
        sx, ex = (int)(center[0]) - img_size // 2, (int)(center[0]) + img_size // 2
        bsy, bsx = 0 - min(0,sy), 0 - min(0,sx)
        bey, bex = ey - min(ey,map_img.shape[0]), ex - min(ex,map_img.shape[1])

        patch = np.zeros((img_size,img_size,3),dtype=np.uint8)
        patch[bsy:img_size-bey,bsx:img_size-bex] = map_img[sy+bsy:ey-bey,sx+bsx:ex-bex]

        center = [(sx+ex)/2,(sy+ey)/2]
        center = np.array([self.map_pos[0] - (width/2 - center[0]) * self.map_gsd / self.meters_lon_degree, 
                        self.map_pos[1] + (height/2 - center[1]) * self.map_gsd / self.meters_lat_degree ])

        return patch, center


    def get_map_patches(self,map_img)->Patches:
        patches = Patches(
                images = [],
                centers = []
            )
        
        width = map_img.shape[1]
        height = map_img.shape[0]

        patch_shape = (int)(self.patch_size / self.map_gsd)

        for y in range(patch_shape//2,map_img.shape[0]-patch_shape//2-1,patch_shape):
            for x in range(patch_shape//2,map_img.shape[1]-patch_shape//2-1,patch_shape):
                patch, center = self.get_patch(map_img,(x,y),width,height)
                patches.images.append(patch)
                patches.centers.append(center)

                for i in range(1,11):
                    for j in range(1,11):
                        patch, center = self.get_patch(map_img,(x+i*patch_shape//20,y+j*patch_shape//20),width,height)
                        patches.images.append(patch)
                        patches.centers.append(center)


        # for y in range(patch_shape//4*3,map_img.shape[0]-patch_shape//4*3-1,patch_shape):
        #     for x in range(patch_shape//4*3,map_img.shape[1]-patch_shape//4*3-1,patch_shape):
        #         patch, center = get_patch(map_img,(x,y),map_pos,width,height)
        #         patches.images.append(patch)
        #         patches.centers.append(center)

        patches.centers = np.array(patches.centers)
        return patches

    def get_closest_patch(self,patches:Patches, prev_pos):
        distances = patches.centers - prev_pos
        distances**=2
        distances = np.sum(distances,axis=1)
        id = np.argmin(distances)
        return patches.images[id], patches.centers[id]

    def patch_localization(self,
                        patches:Patches, 
                        prev_pos, 
                        ground_img,
                        camera_specs : CameraSpecs, 
                        drone_state : DroneState, 
                        points = 100,
                        process_size=512,
                        )->LocOutput:
        patch, center = self.get_closest_patch(patches=patches,prev_pos=prev_pos)
        loc = self.localize(patch,
                    ground_img,
                    points=points,
                    satellite_data=None,
                    center_map_loc=center,
                    process_size=process_size,
                    camera_specs=camera_specs,
                    drone_state=drone_state)
        return loc, patch, center


    def gps_to_relative_pos(self,gps_pos,center_pos,width,height):
        return np.array([(gps_pos[0]-center_pos[0]) * self.meters_lon_degree / self.map_gsd+width/2,
                        (center_pos[1]-gps_pos[1]) * self.meters_lat_degree / self.map_gsd+height/2])

    def absolute_to_relative(self,w,h,abs_pose,map_pos):
        x,y = w/2, h/2
        x += (int)((abs_pose[0]-map_pos[0]) * self.meters_lon_degree / self.map_gsd)
        y += (int)(-(abs_pose[1]-map_pos[1]) * self.meters_lat_degree / self.map_gsd)
        return x,y