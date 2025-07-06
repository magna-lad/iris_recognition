import os
from PIL import Image
import random
import uuid


# import all the images from first the images and then the masks
# do such that each l and r folder has 10 images each for all the cases

TARGET_COUNT = 10
MAX_ANGLE = 20


''' 
        in each folder there will be two folders 
        enter the folder one at a time
        if there are photos< 10, transform them to 10 images
        if there are no images imprt images from the other folder 
        and use them for the same transformations
'''

def path_loader(folder_path):
    image_folders=os.listdir(folder_path)
    image_path=[]
    for folder in image_folders: 
        directory = os.path.join(folder_path,folder)  
        if not os.path.isdir(directory):
            continue  # skip non-folders

        dir_elements= os.listdir(directory) # only reaching the numbered folders
        left_images=[]
        right_images=[]
        for L_R in dir_elements: # reaching the l,r folders
            l_r = (os.path.join(directory,L_R))
            images=os.listdir(l_r)
            if L_R=="L":
                for image in images:
                    left_images.append(os.path.join(l_r,image))
            if L_R=='R':
                for image in images:
                    right_images.append(os.path.join(l_r,image))
        image_path.append([left_images,right_images])
    return image_path


def find_images(folder_path,mask_path):
    ''' 
        in each folder there will be two folders 
        enter the folder one at a time
        save paths of the images
    '''
    image=path_loader(folder_path)
    mask=path_loader(mask_path)
    return  image, mask


# now find the unequal l,r sets and make them equal


def finder(image,mask):
    '''
    finds the list with unequal pairs
    and augments them to have equal number of images==10
    '''
    for idx,i in enumerate(image[:TARGET_COUNT]):
        print(len(i[0]),len(i[1]))
        if len(i[0])!=TARGET_COUNT or len(i[1])!=TARGET_COUNT:
            #print(image[idx])
            maker(image[idx],mask[idx])




def save(image,mask,side,rotated_image,rotated_mask,side_alph):
        parent_dir_image=os.path.dirname(os.path.dirname(image[not side][0]))
        target_dir_image=os.path.join(parent_dir_image,side_alph)
        parent_dir_mask=os.path.dirname(os.path.dirname(mask[not side][0]))
        target_dir_mask=os.path.join(parent_dir_mask,side_alph)
        prefix = "aug"
        unique_id = str(uuid.uuid4())[:8]  # Short random ID
        filename = f"{prefix}_{unique_id}"
        os.makedirs(target_dir_image, exist_ok=True)
        os.makedirs(target_dir_mask, exist_ok=True)
        img_path=os.path.join(target_dir_image,f"{filename}.jpg")
        mask_path=os.path.join(target_dir_mask,f"{filename}.png")
        image[side].append(img_path)
        mask[side].append(mask_path)
        rotated_image.save(img_path)
        rotated_mask.save(mask_path)

def augment_image(image_path, mask_path, flip=False, angle=None):
    pil_image = Image.open(image_path).convert("RGB")
    pil_mask = Image.open(mask_path).convert("L")
    if flip:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        pil_mask = pil_mask.transpose(Image.FLIP_LEFT_RIGHT)
    if angle is None:
        angle = random.randint(-MAX_ANGLE, MAX_ANGLE)
    pil_image = pil_image.rotate(angle)
    pil_mask = pil_mask.rotate(angle)
    return pil_image, pil_mask



def augment_partial_empty_case(image,mask,random_image,random_mask,side,flip):
    '''
    args: random_image,random_mask- singular images
        side- decides if flipping is required
    '''
    if side==0:
            side_alph="L"
    else:
        side_alph="R"
    if flip==0:      
        rotated_image,rotated_mask=augment_image(random_image, random_mask, flip)
        # generate paths for these flipped images and save them in the root folder, left here
        
        save(image,mask,side,rotated_image,rotated_mask,side_alph)
    if flip==1:
        
        rotated_image,rotated_mask=augment_image(random_image, random_mask, flip)
        

        # after rotated image code can be made more modular

        # generate paths for these flipped images and save them in the root folder, left here
        
        save(image,mask,side,rotated_image,rotated_mask,side_alph)

def augment_empty_case(image,mask,side):
    '''
    args:image,mask- list
        side- the side which is empty
    '''
    SIZE=TARGET_COUNT
    if side==0:
        side_alph="L"
    else:
        side_alph="R"
    while len(image[side])<SIZE:
        idx= random.randint(0,len(image[not side])-1)          
        random_image=image[not side][idx]
        random_mask=mask[not side][idx]
        # horizontally flip them
        rotated_image,rotated_mask=augment_image(random_image, random_mask, flip=True)
        

        # after rotation the code can be made more modular

        # generate paths for these flipped images and save them in the root folder, left here

        save(image,mask,side,rotated_image,rotated_mask,side_alph)
        


def maker(image,mask):
    '''
    args: image- lists of [L,R] of various folders
          masks- lists of [L,R] of various folders
    goal: to make the numeber of images in both L and R of all folders equal
    '''
    left,right=0,1
    # if left is empty
    if len(image[left])==0:
        # choose a random (image,masks) from the right folder and make flip them and make augmentation
        augment_empty_case(image,mask,left)
    # if right is empty
    if len(image[right])==0:
        # choose a random (image,masks) from the right folder and make flip them and make augmentation
        augment_empty_case(image,mask,right)

    if len(image[left])<TARGET_COUNT:
        # make a stack of randint of indices of the left and right folders, and fill the less filled one first
        # then fill the more filled one till both have 10 images each
        indices=[] # used as stack
        for i in range(TARGET_COUNT-len(image[left])):
            left_indices = random.randint(0,len(image[left])-1)
            right_indices = random.randint(0,len(image[right])-1)
            indices.append((0,left_indices))
            indices.append((1,right_indices))

        # now make an algo where an index is taken from the stack indices
        # and augmentation of the image is added to the folder

        while len(image[left])<TARGET_COUNT:
            idx=indices.pop()
            side=int(idx[0])
            random_image=image[side][int(idx[1])]
            random_mask=mask[side][int(idx[1])]
            flip=0
            # see if we need to flip the image or not and rotate them and add them to the folders
            if side==0: # no flip
                flip=0
                augment_partial_empty_case(image,mask,random_image,random_mask,side,flip)
            if side==1:
                # horizontally flip them
                flip=1
                augment_partial_empty_case(image,mask,random_image,random_mask,side,flip)
                
    if len(image[right])<TARGET_COUNT:
        # make a stack of randint of indices of the left and right folders, and fill the less filled one first
        # then fill the more filled one till both have 10 images each
        indices=[] # used as stack
        for i in range(TARGET_COUNT-len(image[right])):
            left_indices = random.randint(0,len(image[right])-1)
            right_indices = random.randint(0,len(image[right])-1)
            indices.append((0,left_indices))
            indices.append((1,right_indices))

        # now make an algo where an index is taken from the stack indices
        # and augmentation of the image is added to the folder

        while len(image[right])<TARGET_COUNT:
            idx=indices.pop()
            side=int(idx[0])
            random_image=image[side][int(idx[1])]
            random_mask=mask[side][int(idx[1])]
            # see if we need to flip the image or not and rotate them and add them to the folders
            if side==1: # no flip
                flip=0
                augment_partial_empty_case(image,mask,random_image,random_mask,side,flip)
            if side==1:
                # horizontally flip them
                flip=1
                augment_partial_empty_case(image,mask,random_image,random_mask,side,flip)



image,mask=find_images(r"C:\Users\kound\OneDrive\Desktop\v4_extra\v4",r"C:\Users\kound\OneDrive\Desktop\v4_extra\v4_masks")
finder(image,mask)



