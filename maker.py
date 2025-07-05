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



def aker(image, mask):
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



def augment_empty_case(image,mask,side,flip):
    SIZE=10
    while len(image[side])<SIZE:
            idx= random.randint(0,len(image[not side])-1)          
            random_image=image[not side][idx]
            random_mask=mask[not side][idx]

            # horizontally flip them
            if flip==True:
                pil_image = Image.open(random_image).convert("RGB")
                flipped_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT)


                pil_mask = Image.open(random_mask).convert("L")
                flipped_mask=pil_mask.transpose(Image.FLIP_LEFT_RIGHT)

                angle=random.randint(-20,20)

                rotated_image=flipped_image.rotate(angle)
                rotated_mask=flipped_mask.rotate(angle)

                # generate paths for these flipped images and save them in the root folder, left here
                parent_dir_image=os.path.dirname(os.path.dirname(image[not side][0]))
                target_dir_image=os.path.join(parent_dir_image,'L')

                parent_dir_mask=os.path.dirname(os.path.dirname(mask[not side][0]))
                target_dir_mask=os.path.join(parent_dir_mask,'L')

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
        augment_empty_case(image,mask,left,True)
    # if right is empty
    if len(image[right])==0:
        # choose a random (image,masks) from the right folder and make flip them and make augmentation
        augment_empty_case(image,mask,right,True)

    if len(image[left])<10:
        # make a stack of randint of indices of the left and right folders, and fill the less filled one first
        # then fill the more filled one till both have 10 images each
        indices=[] # used as stack
        for i in range(10-len(image[left])):
            left_indices = random.randint(0,len(image[left])-1)
            right_indices = random.randint(0,len(image[right])-1)
            indices.append(f"0{left_indices}")
            indices.append(f"1{right_indices}")

        # now make an algo where an index is taken from the stack indices
        # and augmentation of the image is added to the folder

        while len(image[left])<10:
            idx=indices.pop()
            side=int(idx[0])
            random_image=image[side][int(idx[1])]
            random_mask=mask[side][int(idx[1])]
            # see if we need to flip the image or not and rotate them and add them to the folders
            if side==0: # no flip
                pil_image = Image.open(random_image).convert("RGB")


                pil_mask = Image.open(random_mask).convert("L")

                angle=random.randint(-20,20)

                rotated_image=pil_image.rotate(angle)
                rotated_mask=pil_mask.rotate(angle)

                # generate paths for these flipped images and save them in the root folder, left here
                parent_dir_image=os.path.dirname(os.path.dirname(image[not side][0]))
                target_dir_image=os.path.join(parent_dir_image,'L')

                parent_dir_mask=os.path.dirname(os.path.dirname(mask[not side][0]))
                target_dir_mask=os.path.join(parent_dir_mask,'L')

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
            if side==1:
                # horizontally flip them
                pil_image = Image.open(random_image).convert("RGB")
                flipped_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT)


                pil_mask = Image.open(random_mask).convert("L")
                flipped_mask=pil_mask.transpose(Image.FLIP_LEFT_RIGHT)

                angle=random.randint(-20,20)

                rotated_image=flipped_image.rotate(angle)
                rotated_mask=flipped_mask.rotate(angle)

                # generate paths for these flipped images and save them in the root folder, left here
                parent_dir_image=os.path.dirname(os.path.dirname(image[not side][0]))
                target_dir_image=os.path.join(parent_dir_image,'L')

                parent_dir_mask=os.path.dirname(os.path.dirname(mask[not side][0]))
                target_dir_mask=os.path.join(parent_dir_mask,'L')

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

    if len(image[right])<10:
        # make a stack of randint of indices of the left and right folders, and fill the less filled one first
        # then fill the more filled one till both have 10 images each
        indices=[] # used as stack
        for i in range(10-len(image[right])):
            left_indices = random.randint(0,len(image[right])-1)
            right_indices = random.randint(0,len(image[right])-1)
            indices.append(f"0{left_indices}")
            indices.append(f"1{right_indices}")

        # now make an algo where an index is taken from the stack indices
        # and augmentation of the image is added to the folder

        while len(image[right])<10:
            idx=indices.pop()
            side=int(idx[0])
            random_image=image[side][int(idx[1])]
            random_mask=mask[side][int(idx[1])]
            # see if we need to flip the image or not and rotate them and add them to the folders
            if side==1: # no rotation
                            # horizontally flip them
                pil_image = Image.open(random_image).convert("RGB")


                pil_mask = Image.open(random_mask).convert("L")

                angle=random.randint(-20,20)

                rotated_image=pil_image.rotate(angle)
                rotated_mask=pil_mask.rotate(angle)

                # generate paths for these flipped images and save them in the root folder, left here
                parent_dir_image=os.path.dirname(os.path.dirname(image[not side][0]))
                target_dir_image=os.path.join(parent_dir_image,'L')

                parent_dir_mask=os.path.dirname(os.path.dirname(mask[not side][0]))
                target_dir_mask=os.path.join(parent_dir_mask,'L')

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
            if side==1:
                # horizontally flip them
                pil_image = Image.open(random_image).convert("RGB")
                flipped_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    
                
                pil_mask = Image.open(random_mask).convert("L")
                flipped_mask=pil_mask.transpose(Image.FLIP_LEFT_RIGHT)
    
                angle=random.randint(-20,20)
    
                rotated_image=flipped_image.rotate(angle)
                rotated_mask=flipped_mask.rotate(angle)
    
                # generate paths for these flipped images and save them in the root folder, left here
                parent_dir_image=os.path.dirname(os.path.dirname(image[not side][0]))
                target_dir_image=os.path.join(parent_dir_image,'L')
    
                parent_dir_mask=os.path.dirname(os.path.dirname(mask[not side][0]))
                target_dir_mask=os.path.join(parent_dir_mask,'L')
    
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



image,mask=find_images(r"C:\Users\kound\OneDrive\Desktop\v4_extra\v4",r"C:\Users\kound\OneDrive\Desktop\v4_extra\v4_masks")
finder(image,mask)



