# -*- coding: utf-8 -*-
"""
Based on paper : http://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf
"""



import cv2
import numpy as np
from scipy import ndimage
from sklearn.feature_extraction import image

PATCH_WIDTH = 9
ALPHA = 255


class Point:
    def __init__(self, x,y):
        self.x = x
        self.y = y

class Gradient:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
    
def extract_pathches(img, patch_width):
    
    patches = image.extract_patches_2d(img, (patch_width, patch_width))
    return patches

class LargeObjectRemoval:
    def __init__(self, original_image, image_mask):
        self.image = original_image
        self.height, self.width = self.image.shape[:2]
        self.mask = image_mask
        self.source_region = None
        self.max_priority_point = None
        self.best_source_patch = None
        self.prev_priority_points = set()
        self._prepare_terms()
    

    def _prepare_terms(self):
        """
        confidence term : target 0, image source 1
        data term : initial to zeros.
        """
        self.confidence = (1-np.float64(self.mask))
        self.data = np.zeros([self.height, self.width]) 
       
    def is_target_empty(self):
        return np.all(self.mask == 0 )
        
    def _compute_contour(self):
        """
        Calculate contour
        """
        h, w = self.mask.shape[:2]
        self.contour = []
        for y in range(h):
            for x in range(w):
                if self.mask[y, x] == 255 and np.isin(0, self.mask[y-1:y+2, x-1:x+2]):
                    self.contour.append(Point(x, y))
        # once contour is computed, corresponding normal will also need to be updated
        self._compute_contour_normal()

    def get_image_patch(self, image, region):
        (left_x, left_y), (right_x, right_y) = region
        return image[left_y:right_y, left_x:right_x]
    

    def _prepare_patch_region(self, point):
        HALF_PATCH_WIDTH = (PATCH_WIDTH)//2
        center_x, center_y = point.x, point.y
        min_x = max(center_x - HALF_PATCH_WIDTH, 0)
        max_x = min(center_x + HALF_PATCH_WIDTH + 1, self.width - 1)
        min_y = max(center_y - HALF_PATCH_WIDTH, 0)
        max_y = min(center_y + HALF_PATCH_WIDTH + 1, self.height - 1)
        upper_left = (min_x, min_y)
        lower_right = (max_x, max_y)
        return upper_left, lower_right
   
    def C_p(self, point):
        """
        Confiidence value of a given central point
        """
        p_x, p_y = point.x, point.y
        # print('before update',self.confidence[p_y, p_x] )
        (left_x, left_y), (right_x, right_y) = self._prepare_patch_region(point)
        conf_sum = 0
        for y in range(left_y, right_y + 1):
            for x in range(left_x, right_x + 1):
                if self.mask[y, x] == 0:
                    conf_sum += self.confidence[y, x]
        confidence = conf_sum / ((right_x-left_x+1) * (right_y-left_y+1))
        self.confidence[p_y, p_x] = confidence
        return confidence

    def D_p(self, point):
        """
        It prepares data term
        """
        col, row = point.x, point.y
        gx = self.gradient.dx[row, col]
        gy = self.gradient.dy[row, col]
        # normalX, normalY = dy, - dx 
        isophote = np.array([gy, -gx])
        nx = self.normal.dx[row,col]
        ny = self.normal.dy[row,col]

        normal = np.array([nx, ny])
        return abs( isophote @ normal ) / ALPHA


    def _calculate_priority_for_apoint(self, point):
        """
            Priority of a given patch given it's central point
        """
        col, row = point.x, point.y
        C_p = self.C_p(point)
        D_p = self.D_p(point)

        return C_p * D_p

    def _compute_priorities(self):
        max_priority_point = None
        max_priority = 0
        for point in self.contour:
            col, row = point.x, point.y
            priority = self._calculate_priority_for_apoint(point)
            if priority > max_priority and point not in self.prev_priority_points:
                max_priority = priority
                self.max_priority_point = Point(col,row)
                self.prev_priority_points.add(point)

    def _compute_image_gradient(self):
        img_gray = np.array(cv2.cvtColor(self.image,
            cv2.COLOR_BGR2GRAY), dtype=np.float64)
        gx = ndimage.sobel(img_gray, axis=1) # horizontal derivative
        gy = ndimage.sobel(img_gray, axis=0) # vertical derivativeass

        self.gradient = Gradient(gx, gy)

    def _compute_contour_normal(self):

        nx = ndimage.sobel(self.mask.astype(float), axis=1) # horizontal derivative
        ny = ndimage.sobel(self.mask.astype(float), axis=0) # vertical derivative

        length = np.sqrt(nx ** 2 + ny ** 2)
        nx = nx / length
        nx[ nx == inf ] = 0; # handle NaN and Inf
        ny = ny / length 
        ny[ ny== inf ]=0
        nx[np.isnan(nx)] = 0
        ny[np.isnan(ny)] = 0
        self.normal = Gradient(nx, ny)
    
    def get_rgb(self, data):
        height, width = data.shape
        return data.reshape(height, width, 1).repeat(3, axis=2)
    
    def ssd(self, source, target, mask_3d):
      
        diff = source - target
        return np.sum(np.square(np.multiply(diff, mask_3d)))
    
    def calculate_ssd(self, p, q):
        """
            Calculate the ssd between target p and source q
        """
        img_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        half_width = PATCH_WIDTH//2
        source_patch_region = self._prepare_patch_region(q)
        source = self.get_image_patch(img_lab, source_patch_region)
        mask_patch = self.mask[p.y-half_width:p.y+half_width, p.x-half_width:p.x+half_width]
        mask_3d = mask_patch.reshape(source.shape[0], source.shape[1],1).repeat(3, axis=2)
        mask_3d[mask_3d > 100] = 255
        mask_3d[mask_3d < 100] = 1
        mask_3d[mask_3d == 255] = 0
        target_patch_region = self._prepare_patch_region(p)
        target = self.get_image_patch(img_lab, target_patch_region)
     
        diff = source - target
        return np.sum(np.square(np.multiply(diff, mask_3d)))
    
    def _compute_best_source_patch(self):
      
        img_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
    
        min_patch_diff = float("inf")
        patches = extract_pathches(img_lab, PATCH_WIDTH)
        half_width = PATCH_WIDTH//2
        mod_run_w = (self.width - PATCH_WIDTH) + 1
        mod_run_h = (self.height - PATCH_WIDTH) + 1
        p = self.max_priority_point
        target_patch_region = self._prepare_patch_region(p)
        target = self.get_image_patch(img_lab, target_patch_region)
        mask_patch = self.mask[p.y-half_width:p.y+half_width + 1, p.x-half_width:p.x+half_width + 1]
        try:
            mask_3d = mask_patch.reshape(target.shape[0], target.shape[1],1).repeat(3, axis=2)
        except:
            return 
        mask_3d[mask_3d > 100] = 255
        mask_3d[mask_3d < 100] = 1
        mask_3d[mask_3d == 255] = 0
        
        current_ind = 0
        for ind, source in enumerate(patches):
            diff = self.ssd(source, target, mask_3d)
            if diff < min_patch_diff:
                min_patch_diff = diff
                x = ind % mod_run_w + 1
                y = ind // mod_run_h + 1
                self.best_source_patch = Point(x,y)
                current_ind = ind

    def _update_target_region(self):
        """
            Copy data from source to target region
        """
        half_patch = PATCH_WIDTH//2
       
        p = (self.max_priority_point.y, self.max_priority_point.x)
        q = (self.best_source_patch.y, self.best_source_patch.x)
        for d_y in range(-half_patch, half_patch+1):
          for d_x in range(-half_patch, half_patch+1):
              yy = q[0] - d_y
              xx = q[1] - d_x
              if xx < self.width and yy < self.height:
                  self.image[p[0]-d_y, p[1]-d_x] = self.image[yy, xx]
          
    def _update_confidence_and_mask(self):
        """
            Once the priority point is found, the mask and confidence
            score will be updated with setting to 1.
        """
        half_patch = PATCH_WIDTH//2
        x,y = self.max_priority_point.x, self.max_priority_point.y
        left_x = x - half_patch
        right_x = x + half_patch + 1
        left_y = y - half_patch
        right_y = y + half_patch + 1
        row = col = 0

        for y in range(left_y, right_y):

            for x in range(left_x, right_x):
                # since the target patch is filled with corresponding patch
                # from source it should be updated with 1.
                if x < self.width and y < self.height:
                  self.confidence[y,x] = 1
                  # also since the the contour is moved inward, in the corresponding
                  # contour it should be set to 0
                  self.mask[y, x] = 0
        
    def run(self):
        self._compute_image_gradient()
        
        i = 0
        while not self.is_target_empty():
            self._compute_contour()
            self._compute_priorities()
            self._compute_best_source_patch()

            self._update_target_region()
            self._update_confidence_and_mask()             
            file_name = 'run_{}.jpg'.format(i)
            mask_name = 'mask_{}.jpg'.format(i)
            print('current run ....', i)
            cv2.imwrite(file_name, self.image)
            cv2.imwrite(mask_name, self.mask)
            i += 1


 
if __name__ == '__main__' :
 
    # Read image
    im = cv2.imread("image7.jpg")

    frame = cv2.imread("mask7.jpg")
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    obj = LargeObjectRemoval(im, gray_img)
    obj.run()

