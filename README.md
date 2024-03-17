## Supress Warnings
In this section, we suppress any warnings that may arise during the execution of the code.

```python
import warnings
warnings.filterwarnings('ignore')
```

This code snippet prevents warnings from being displayed, which can help maintain a clean output and avoid distractions.

## GeoTiff Images

Here, we import libraries for working with GeoTiff images, which are commonly used in geospatial applications. These will be used to extract pre-event and post-event images of San Juan.

```Python
import rasterio
from osgeo import gdal
```

## data-visualization

In this section, we are importing the libraries needed to visualize data, such as matplotlib for plots and PIL for image processing.
```python
from matplotlib import pyplot as plt
import matplotlib.image as img
from matplotlib.pyplot import figure
from PIL import Image #image processing
```

## Model Construction

Here we are importing libraries related to building the model. In particular, we are using the ultralytics package to implement a YOLO model and the labelme2yolo function to convert labels from Labelme formats to YOLO format.

```Python
import ultralytics
from ultralytics import YOLO
import labelme2yolo
```
## Others

In this block we are importing other libraries necessary for code execution, such as OS for system operations, shutil for file and folder operations, and zipfile for working with zip files.
```Python
import os
import shutil
import zipfile
```

We are also using %matplotlib inline to ensure that the plots are displayed inline within the notebook.

```Python
%matplotlib inline
```

## Download Pre-Event and Post-Event Images of San Juan
In this section, we download both the pre-event and post-event images of San Juan from the specified URLs.

### Pre-Event Image
To download the pre-event image of San Juan, we use the following command:

```bash
!wget https://challenge.ey.com/api/v1/storage/admin-files/Pre_Event_San_Juan.tif -O Pre_Event_San_Juan.tif
```

This command utilizes the `wget` utility to download the pre-event image of San Juan from the provided URL. The image is saved as `Pre_Event_San_Juan.tif`.

### Post-Event Image

For the post-event image of San Juan, we execute the following command:

```bash
!wget https://challenge.ey.com/api/v1/storage/admin-files/Post_Event_San_Juan.tif -O Post_Event_San_Juan.tif
```

Similarly, this command employs `wget` to download the post-event image of San Juan from the specified URL. The image is saved as `Post_Event_San_Juan.tif`.

## Pre-Event and Post-Event Images of San Juan
In this section, we define the file paths for both the pre-event and post-event images of San Juan.

### Pre-Event Image
The file path for the pre-event image of San Juan is defined as follows:

```python
pre_event_image = './Pre_Event_San_Juan.tif'
```

This variable `pre_event_image` holds the file path to the pre-event image file, 'Pre_Event_San_Juan.tif'.

### Post-Event Image

Similarly, the file path for the post-event image of San Juan is specified as:

```python
post_event_image ='./Post_Event_San_Juan.tif'
```

The variable `post_event_image` contains the file path to the post-event image file, 'Post_Event_San_Juan.tif'.

## Generate Tiles from Input Image

This function generates tiles from an input image and saves them as separate TIFF images.

### Function Signature

```python
def generate_tiles(input_file, output_dir, grid_x, grid_y):
    """
    Generate tiles from a raster image.

    Parameters:
        input_file (str): The path to the input raster image file.
        output_dir (str): The directory where the generated tiles will be saved.
        grid_x (int): The width of each tile in pixels.
        grid_y (int): The height of each tile in pixels.

    Returns:
        None

    Generates tiles from the input raster image specified by `input_file`. Each tile has dimensions `grid_x` by `grid_y`.
    The generated tiles are saved as separate TIFF images in the `output_dir`.

    Note:
        This function requires the GDAL library to be installed.

    Example:
        generate_tiles('input_image.tif', 'output_tiles/', 256, 256)
    """
    # Open the input raster image
    ds = gdal.Open(input_file)

    # Get image size and number of bands
    width = ds.RasterXSize
    height = ds.RasterYSize
    num_bands = ds.RasterCount

    # Calculate number of tiles in each dimension
    num_tiles_x = (width // grid_x)
    num_tiles_y = (height // grid_y)

    print(f"Total number of tiles: {num_tiles_x * num_tiles_y}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each tile and save as a separate TIFF image
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x_offset = i * grid_x
            y_offset = j * grid_y

            # Calculate tile width and height
            tile_width = min(grid_x, width - x_offset)
            tile_height = min(grid_y, height - y_offset)

            # Read data for each band in the tile
            tile = []
            for band in range(1, num_bands + 1):
                tile_data = ds.GetRasterBand(band).ReadAsArray(x_offset, y_offset, tile_width, tile_height)
                tile.append(tile_data)

            # Create output filename
            output_file = os.path.join(output_dir, f"tile_{i}_{j}.tif")

            # Create an output TIFF file with compression and tiling
            driver = gdal.GetDriverByName("GTiff")
            options = ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'TILED=YES']
            out_ds = driver.Create(output_file, tile_width, tile_height, num_bands,
                                   ds.GetRasterBand(1).DataType, options=options)

            # Set the geotransform
            geotransform = list(ds.GetGeoTransform())
            geotransform[0] = geotransform[0] + x_offset * geotransform[1]
            geotransform[3] = geotransform[3] + y_offset * geotransform[5]
            out_ds.SetGeoTransform(tuple(geotransform))

            # Set the projection
            out_ds.SetProjection(ds.GetProjection())

            # Write each band to the output file
            for band in range(1, num_bands + 1):
                out_band = out_ds.GetRasterBand(band)
                out_band.WriteArray(tile[band - 1])

            # Close the output file
            out_ds = None

    print("Tiles generation completed.")

```

### Parameters:

- `input_file` (str): The path to the input image file.
- `output_dir` (str): The directory where the generated tiles will be saved.
- `grid_x` (int): The number of tiles along the x-axis.
- `grid_y` (int): The number of tiles along the y-axis.

### Returns:

This function does not return any value. It generates and saves the tiles as separate TIFF images.

### Details:

1. **Open Input Image**: Opens the input image file using GDAL.
2. **Get Image Size and Bands**: Retrieves the size of the image (width and height) and the number of bands.
3. **Calculate Number of Tiles**: Determines the number of tiles to be generated in each dimension based on the specified grid size.
4. **Create Output Directory**: Creates the output directory if it does not already exist.
5. **Generate Tiles**: Iterates over each tile in the grid, extracts the corresponding portion of the image, and saves it as a separate TIFF file.
6. **Set GeoTransform and Projection**: Sets the geotransform and projection information for each tile based on the input image.
7. **Write Tile Data**: Writes the data for each band of the tile to the output file.
8. **Close Output File**: Closes the output file after writing all bands.
9. **Completion Message**: Prints a message indicating the completion of tile generation.

### Generate tiles from Pre_Event_San_Juan

```Python
# Paths to the input raster image file and the directory where the generated tiles will be saved
input_file = "./Pre_Event_San_Juan.tif"
output_dir = "./Pre_Event_Grids_In_TIFF"

# Dimensions of each tile in pixels
grid_x = 512
grid_y = 512

# Generate tiles from the input raster image
generate_tiles(input_file, output_dir, grid_x, grid_y)

```

This code uses the generate_tiles function to split the raster image specified by input_file into tiles of size grid_x by grid_y and save them as TIFF files in the output_dir directory.

### Generate tiles from Post_Event_San_Juan

```Python
# Paths to the input raster image file and the directory where the generated tiles will be saved
input_file = "./Post_Event_San_Juan.tif"
output_dir = "./Post_Event_Grids_In_TIFF"

# Dimensions of each tile in pixels
grid_x = 512
grid_y = 512

# Generate tiles from the input raster image
generate_tiles(input_file, output_dir, grid_x, grid_y)

```

This code uses the generate_tiles function to split the raster image specified by input_file into tiles of size grid_x by grid_y and save them as TIFF files in the output_dir directory.

## Renaming the Files

Once we have the tiles we can label the images along with their processing

This process involves both pre-event and post-event images to identify residences that received damages post-event. Consequently, the generated images or tiles need to be renamed using the `imageCount` format. To achieve this, a different code is used compared to the one provided by the platform. Changing the tile names ensures that the order of the generated files remains consistent. This prevents difficulty in identifying pre-event and post-event images during labeling. A custom code is programmed to rename the tiles in such a way that the order remains intact. For example, if we search for `image1`, it corresponds to the same geographical position in both the pre_event and post_event folders. This streamlines the labeling process as we can easily identify if a residence was damaged after the event. Additionally, QGIS is employed to locate these residences and verify their bounding box to extract the most accurate label possible.

```Python

def list_files(path):
	# Get the list of files in the path
	files = os.listdir(path)
	# Return the list of files
	return files
	
def copy_files(Opath, Fpath, newName):
  """
  This function copies a file from one path to another with a new name.
  Parameters:
    Opath: The path to the file you want to copy.
    Fpath: The path to the folder where you want to copy the file.
    newName: The new file name.
  Return:
    None
  """
  # Copy the file
  shutil.copyfile(Opath, os.path.join(Fpath, newName))
  # Ruta de la carpeta que se desea listar

path = "./dataset/images"
# search for files with that name in another folder
path1 = "./dataset/Annots"
# Get list of files
IFiles = list_files(path)
# Print file list
for f in IFiles:
  print(f[:-4])
  
# Path of the file you want to copy
pathA = "./dataset/Annots/"
pathI = "./dataset/images/"

# Path of the folder where you want to copy the file
pathAF = "./dataset/AnnotsF/"
pathIF = "./dataset/imagesF/"
count = 1

for archive in archives:
    print(archive)
    # Copy the file
    copy_files(pathA+archive[:-4]+".json", pathAF, "image"+str(count)+".json")
    copy_files(pathI+archive, pathIF, "image"+str(count)+".jpg")
    count += 1
print("The file has been copied successfully.")
```

Next, the photos are annotated using LabelMe. We have four classes to label that are the following:
-   **Damaged Commercial Building:** This class refers to buildings used exclusively for commercial purposes that exhibit structural damage. To identify such buildings, we examine the space around them, looking for indicators like parking spaces or a moderate presence of people, which could suggest commercial use. Damage is assessed primarily on the roof, where visible breaks or the presence of blue mesh/shade cloth indicate damage.
-   **Damaged Residential Building:** This class applies to buildings used solely for domestic purposes that show structural damage. Identification involves examining the surrounding area for characteristics - - Undamaged Commercial Building: This class refers to commercially used buildings without visible structural damage. Identification criteria include the presence of parking or some level of people activity around the building, suggesting commercial use, without any apparent damage.
-   **Undamaged Residential Building:** This class denotes buildings used for domestic purposes without visible structural damage. To identify these buildings, we look for features like significant land or adjacent houses with similar structures, indicative of residential use, and the absence of apparent damage.

As mentioned previously, we utilize QGIS software because it allows us to open .tiff images. We then import the Pre-Event and Post-Event images as well as the building footprint as layers. These image layers enable us to observe the changes in buildings pre- and post-disaster at the same location, while the footprint aids in visualizing the building's shape, making it easier to label within the boxes. This step is one of the most important, as it was crucial in verifying the accuracy of our labels.
## Generation of Training and Test Data with Configuration File

In this section, we will generate the necessary training and testing data for our object detection model. We will also create a configuration file that specifies the location of the data files.
### Training and Test Data Generation

We will use the labelme2yolo tool to convert LabelMe format annotations to YOLO format, which is required by our object detection model. This tool takes as input a directory containing the annotation JSON files and produces the text files necessary for training and testing the model.

```Bash
!labelme2yolo --json_dir ./LABEL
```

This command will run the annotation conversion in the `./LABEL` directory and generate the data files needed for training and testing the model.
## Model Training

In this section, we will load a pre-trained model and perform additional training as necessary for our specific object detection task.

### Model Load

Before starting training, we will load a pre-trained model. We will use the`YOLO` class from the Ultralytics library to load the model. We will specify the pre-trained model file as an argument.

```Python
# Load the model
model = YOLO('yolov8m-obb.pt')
```

In addition to trying to implement the YOLO model, we also explored the implementation of the Mask R-CNN model. After extensive research, we found that this model provided the best results in computer vision focused on satellite images. However, because the model runs on TensorFlow 1.3, we were unable to complete its implementation in the time available. Although we managed to make progress in the implementation, we fell short in time to make it fully functional. Notably, this model performs exceptionally well due to our ability to use ResNet101 as the backbone of the model.

### Viewing Model Information (optional)

Optionally, we can display information about the loaded model. This may include details such as the model architecture, the number of parameters, and other relevant characteristics.

```Python
# Display model information
model.info()
```

This command displays detailed information about the loaded model, which can be helpful in understanding its structure and configuration.

From this point, we can proceed with additional training of the model, adjusting the hyperparameters as necessary for our specific task.

During the competition, different versions of yolo were tested along with the modification of different hyperparameters, such as:

- `mixup`: Mix two images and their labels, creating a composite image. Enhances the model's generalization ability by introducing label noise and visual variability.
- `erasing`: Randomly erase a part of the image during classification training, encouraging the model to focus on less obvious features for recognition.
- `translate`: Translate the image horizontally and vertically by a fraction of the image size, which helps in learning to detect partially visible objects.
- `scale`: Scale the image by a gain factor, simulating objects at different distances from the camera.
- `flipud`: Flip the image with the specified probability, increasing data variability without affecting object features.
- `fliplr`: Flip the image from left to right with the specified probability, useful for learning symmetric objects and increasing dataset diversity.
- `mosaic`: Combine four training images into one, simulating different scene compositions and object interactions. Highly effective for understanding complex scenes
- `degrees`: Rotate the image randomly within the specified degree range, enhancing the model's ability to recognize objects in various orientations.
- `translate`: Translate the image horizontally and vertically by a fraction of the image size, which helps in learning to detect partially visible objects.

To modify these parameters we must use them within the train function and give them a number which is in the range of the respective parameter.

## Model Training Results

Once we have completed training the model, we can examine the results obtained, such as performance metrics and any other relevant details about the training process.

### Model Training

To train the model, we use the `train` method of the model instance, specifying the training data, the number of epochs, and the desired image size.

```Python
results = model.train(data='./dataset.yaml', epochs=85, imgsz=640)
```

This command starts training the model with the following parameters:

- `data`: Specifies the location of the YAML file that contains information about the training data.
- `epochs`: Specifies the number of epochs for training.
- `imgsz`: Specifies the size of the image to use during training.

In addition to these parameters, many others can be modified, which were tested throughout the challenge. Among these is the optimizer which AdamW was chosen since it gave the best result, it is also the one that comes by default so there is no need to edit it in the model parameters.

Another parameter which was tried to modify is Learning rate, in yolo we have 2 types of modifiable learning rate. One is the `lr0` which is the initial learning rate which influences the speed with which the weights of the model are updated and the another is the `lrf` which is the Final Learning Rate used to adjust the learning rate over time.

### Training Results

After completing the training, we can examine the results obtained, such as performance metrics and any other relevant details.

### Show training results

```Python
print(results)
```

This command displays the training results, which may include metrics such as loss, precision, convergence speed, and more.

It is important to review these results to evaluate model performance and determine if additional adjustments to training or model settings are needed.

## Model Evaluation

After training the model, it is crucial to evaluate its performance using test data to understand its ability to perform the object detection task. In this section, we will examine how to evaluate the model and visualize the results obtained.

### Viewing Model Evaluation Results

To visualize the results of the model evaluation, we will display an image that contains performance metrics such as precision, recall, and other relevant measures.

In the runs folder there are a large number of visualizations of the training results such as graphs and an Excel table with the metrics. It also has a .yaml which contains the configurations used in the model.

```Python
from matplotlib import pyplot as plt
import matplotlib.image as img
from matplotlib.pyplot import figure

# Create a shape to display the image
figure(figsize=(15, 10), dpi=80)

# Read the image of the evaluation results
results = img.imread('runs/obb/train/results.png')

# Show the image of the evaluation results
plt.imshow(results)
````

This code creates a custom-sized figure and displays the image containing the model evaluation results.

Make sure to provide the correct path to the results image as it changes depending on how many times we train our model.

It is important to visually review these results to understand the model's performance on the object detection task and determine if additional adjustments to the model or training settings are needed.

### Viewing the Model Confusion Matrix

The confusion matrix is ​​a useful tool to evaluate the performance of a classification model. In this section, we will visualize the confusion matrix generated during the model evaluation.

### Visualization of the Confusion Matrix

We will use the matplotlib library to display the confusion matrix as an image.

```Python
from matplotlib import pyplot as plt
import matplotlib.image as img
from matplotlib.pyplot import figure

# Create a figure to show the confusion matrix
figure(figsize=(20, 15), dpi=80)

# Read the image of the confusion matrix
cf = img.imread('runs/obb/train/confusion_matrix.png')

# Show the image of the confusion matrix
plt.imshow(cf)
```

This code creates a custom-sized figure and displays the image containing the model's confusion matrix.

Make sure to provide the correct path to the results image as it changes depending on how many times we train our model.

Visualization of the confusion matrix provides an overview of the performance of the model in terms of object classification, which can be useful in identifying possible areas of improvement in the model.

## Download submission images from platform

First we will download the .ZIP `challenge 1 submission_images` from the platform, this ZIP contains the images that will be used to validate our data and therefore to send the submission

### Function to Unzip a Folder

We will use an `unzip_folder` function to unzip a ZIP file to a specific directory.

```Python
def unzip_folder(zip_filepath, dest_dir):
    """
    Decompresses a ZIP file to a specific directory.

    Parámetros:
    - zip_filepath (str): Path of the ZIP file to be decompressed.
    - dest_dir (str): Destination directory for extraction.
    """
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f'The zip file {zip_filepath} has been extracted to the directory {dest_dir}')
```

This function takes the path of the ZIP file to be unzipped and the destination directory as arguments and extracts all the contents of the ZIP file to the specified directory.

### Downloading and Extracting submission Images

After defining the decompression function, we can download and extract the submission images from the platform.

```Python
# Specify the path of the Subission ZIP file
submission_zip = './challenge_1_submission_images.zip'

# Specify the destination directory for extraction
submission_directory = './challenge_1_submission_images'

# Unzip the ZIP file
unzip_folder(submission_zip, submission_directory)
```

This code unzips the submission images to a specific directory.

Make sure you provide the correct paths for both the sending ZIP file and the destination directory.

## Making Predictions on Submission Data

Once we have trained our model and loaded the weights of the best trained model, we can use it to make predictions on the submission data.

### Load the Trained Model

First, we need to load the trained model using the best model weights obtained during training. This can be done using the YOLO class from the Ultralytics library.

```Python
# Load the trained model with the best weights
model = YOLO('runs/obb/train/weights/best.pt')
```

This command loads the trained model using the best model weights saved in the directory `runs/obb/train/weights/`.

Once the model is loaded, we can use it to make predictions on the Submission data.

## Decoding Predictions and Generating Output Text Files

After making predictions on the delivery data using the trained model, we need to decode the predictions and generate output text files that follow the format specified in the YAML file.

### Decoding the Predictions

First, we need to define a dictionary that maps the predicted classes to their corresponding names according to the order specified in the YAML file.

```Python
decoding_of_predictions = {
    0: 'damagedcommercialbuilding',
    1: 'damagedresidentialbuilding',
    2: 'undamagedcommercialbuilding',
    3: 'undamagedresidentialbuilding'
}
```

This dictionary will help us decode the predicted classes into their corresponding names.

Be sure to check if the YAML file generated by yolo has the tags in the same order specified in the `decoding_of_predictions` dictionary

### Generation of Output Text Files

Then, we iterate over each file in the input directory, perform predictions on each image, and generate an output text file containing the detections information.

```Python
# Decoding according to the .yaml file class names order
directory = 'challenge_1_submission_images/Submission data'

# Directory to store outputs
results_directory = 'Validation_Data_Results'

# Create submission directory if it doesn't exist
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Loop through each file in the directory
for filename in os.listdir(directory):
    #Check if the current object is a file and ends with .jpeg
    if os.path.isfile(os.path.join(directory, filename)) and filename.lower().endswith('.jpg'):
        # Perform operations on the file
        file_path = os.path.join(directory, filename)
        print(file_path)
        print("Making a prediction on ", filename)
        results = model.predict(file_path, save=True, iou=0.5, save_txt=True, conf=0.25)

        # Decode the predictions
        for r in results:
            conf_list = r.obb.conf.cpu().numpy().tolist()
            clss_list = r.obb.cls.cpu().numpy().tolist()
            original_list = clss_list
            updated_list = []
            for element in original_list:
	            updated_list.append(decoding_of_predictions[int(element)])

        bounding_boxes = r.obb.xyxy.cpu().numpy()
        confidences = conf_list
        class_names = updated_list

        # Check if bounding boxes, confidences and class names match
        if len(bounding_boxes) != len(confidences) or len(bounding_boxes) != len(class_names):
            print("Error: Number of bounding boxes, confidences, and class names should be the same.")
            continue

        # Creating a new .txt file for each image in the submission_directory
        text_file_name = os.path.splitext(filename)[0]
        with open(os.path.join(results_directory, f"{text_file_name}.txt"), "w") as file:
            for i in range(len(bounding_boxes)):
	            # Get coordinates of each bounding box
                left, top, right, bottom = bounding_boxes[i]
                # Write content to file in desired format
                file.write(f"{class_names[i]} {confidences[i]} {left} {top} {right} {bottom}\n")
        print("Output files generated successfully.")

```

This code generates a text file for each image in the input directory. Each file contains the object detection predictions in the format specified in the YAML file.

## Creating a ZIP File from a Directory

After completing the predictions on the shipping data and getting the results, we can package these results into a ZIP file to upload to the platform and verify the accuracy.

### Define the source directory and destination of the ZIP File

Before creating the ZIP file, we first need to define the source directory containing the results and the name and location of the ZIP file to be created.

```Python
# Source directory containing the results
source_dir = results_directory

# Destination path where the ZIP file will be created
destination_zip = 'submission'
```

- `results_directory`: Represents the directory that contains the results that we want to package in the ZIP file
- `submission`: Is the name we want to give to the resulting ZIP file.

### Create a ZIP File from the Directory

Once the source directory and location of the ZIP file have been defined, we can create the ZIP file using Python's `shutil.make_archive()` function.

```Python
# Create a ZIP file from the directory
shutil.make_archive(destination_zip, 'zip', source_dir)
print(f"Directory {source_dir} has been successfully zipped into {destination_zip}.")
```

This code packages all files and subdirectories of the source directory into a ZIP file at the specified location. The resulting ZIP file will have the same name as the value provided in `destination_zip` with the extension `.zip`.

Once the ZIP file has been created, it will be ready to submit to the platform and verify the accuracy of our model.