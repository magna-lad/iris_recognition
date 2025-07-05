import os
from PIL import Image
import random
import uuid


# import all the images from first the images and then the masks
# do such that each l and r folder has 10 images each for all the cases

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
    for idx,i in enumerate(image[:10]):
        print(len(i[0]),len(i[1]))
        if len(i[0])!=10 or len(i[1])!=10:
            #print(image[idx])
            maker(image[idx],mask[idx])



def augment_pair(image_path, mask_path, flip, rotate_angles=[-15, -10, -5, 5, 10, 15]):
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    # Apply horizontal flip if specified
    if flip in ["left", "right"]:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Apply the same random rotation
    angle = random.choice(rotate_angles)
    image = image.rotate(angle)
    mask = mask.rotate(angle)

    return image, mask  # PIL objects


def save_augmented(image, mask, img_dir, msk_dir, prefix="aug"):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    unique_id = str(uuid.uuid4())[:8]
    img_filename = f"{prefix}_{unique_id}.png"
    msk_filename = f"{prefix}_{unique_id}_mask.png"

    img_path = os.path.join(img_dir, img_filename)
    msk_path = os.path.join(msk_dir, msk_filename)

    image.save(img_path)
    mask.save(msk_path)

    return img_path, msk_path



def maker(image, mask):
    '''
    image: [list_of_left_image_paths, list_of_right_image_paths]
    mask : [list_of_left_mask_paths, list_of_right_mask_paths]
    Saves augmented images in original L/R directories.
    '''
    TARGET_COUNT = 10
    left, right = 0, 1

    def fill_to_10(img_list, msk_list, src_imgs, src_msks, flip_label, img_dir, msk_dir, prefix):
        while len(img_list) < TARGET_COUNT:
            idx = random.randint(0, len(src_imgs) - 1)
            img_path = src_imgs[idx]
            msk_path = src_msks[idx]

            augmented_img, augmented_msk = augment_pair(img_path, msk_path, flip_label)
            new_img_path, new_msk_path = save_augmented(
                augmented_img, augmented_msk, img_dir, msk_dir, prefix=prefix
            )

            img_list.append(new_img_path)
            msk_list.append(new_msk_path)

    # Detect base L/R directories from first image
    if image[left]:
        l_dir = os.path.dirname(image[left][0])
        l_mask_dir = os.path.dirname(mask[left][0])
    if image[right]:
        r_dir = os.path.dirname(image[right][0])
        r_mask_dir = os.path.dirname(mask[right][0])

    if len(image[left]) == 0:
        fill_to_10(image[left], mask[left], image[right], mask[right], "right", r_dir, r_mask_dir, prefix="L")
    elif len(image[right]) == 0:
        fill_to_10(image[right], mask[right], image[left], mask[left], "left", l_dir, l_mask_dir, prefix="R")

    if len(image[left]) < TARGET_COUNT:
        fill_to_10(image[left], mask[left], image[left], mask[left], "left", l_dir, l_mask_dir, prefix="L")

    if len(image[right]) < TARGET_COUNT:
        fill_to_10(image[right], mask[right], image[right], mask[right], "right", r_dir, r_mask_dir, prefix="R")




image,mask=find_images(r"C:\Users\kound\OneDrive\Desktop\v4_extra\v4",r"C:\Users\kound\OneDrive\Desktop\v4_extra\v4_masks")
finder(image,mask)



