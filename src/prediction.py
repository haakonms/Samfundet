import numpy as np
from PIL import Image

def pred_to_submission_strings(y_test):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def pred_to_submission(submission_filename, y_test):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def prediction_to_submission(filename, y_submit):
    with open(filename, 'w') as f:
        f.write('id,prediction\n')
        #for i in range(72200):
        i=0;
        for j in range(1,50+1):
          for k in range(0,593,16):
            for l in range(0,593,16): 
              strj = ''
            
              if len(str(j))<2:
                strj='00'
              elif len(str(j))==2:
                  strj='0'

              text = strj + str(j) + '_' + str(k) + '_' + str(l) + ',' + str(y_submit[i])
              f.write(text)
              f.write('\n')
              i=i+1;

def prediction_to_submission2(filename, y_submit):
    with open(filename, 'w') as f:
        f.write('id,prediction\n')
        #for i in range(72200):
        print(y_submit.shape)
        i=0;
        for j in range(1,50+1):
          for k in range(0,593,16):
            for l in range(0,593,16): 
              strj = ''
            
              if len(str(j))<2:
                strj='00'
              elif len(str(j))==2:
                  strj='0'

              text = strj + str(j) + '_' + str(k) + '_' + str(l) + ',' + str(int(y_submit[i,1]))
              f.write(text)
              f.write('\n')
              i=i+1;