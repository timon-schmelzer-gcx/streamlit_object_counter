import typing

import cv2
import numpy as np
import streamlit as st

from utils.configs import IMAGES, DetectionConfig, RunningModes
from utils.configs import default_config
from utils.configs import birds_config
from utils.configs import birds2_config
from utils.configs import keyboard_config
from utils.configs import dots_config

@st.experimental_singleton
def load_image(path: str):
    '''Loads input image from path using opencv.'''
    return cv2.imread(path)

def convert_color(img: np.array):
    '''Sets the correct colors for matplotlib.'''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def convert_gray(img: np.array):
    '''Preprocess image and returns gray version of it.'''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img

def find_contours(img: np.array,
                  thresh_val: int=150,
                  thresh_maxval: int=255,
                  thresh_type: int=1,
                  contour_mode: int=1,
                  contour_method: int=2) -> np.array:
    '''Detect contours and return results.'''
    img = convert_gray(img)
    _, thresh = cv2.threshold(img,
                              thresh_val,
                              thresh_maxval,
                              thresh_type)
    contours, _ = cv2.findContours(thresh, contour_mode, contour_method)

    return contours

def draw_contours(img: np.array,
                  contours: np.array,
                  min_contour_length: int=-9999,
                  max_contour_length: int=9999,
                  contour_index: int=-1,
                  contour_color: typing.Tuple=(255, 0, 0, 1),
                  contour_thickness: int=0) -> typing.Tuple[int, typing.Tuple]:
    '''Add contours if they fullfill the given criteria.'''
    n_contours = 0
    for contour in contours:
        if len(contour) >= min_contour_length and len(contour) <= max_contour_length:
            cv2.drawContours(img,
                             [contour],
                             contour_index,
                             contour_color,
                             contour_thickness)
            n_contours += 1

    return n_contours, img

def format_uploaded_image(uploaded_file: typing.Any) -> np.array:
    '''Tries to format uploaded file into an image.'''
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        return img

def create_sidebar() -> typing.Tuple[np.array, DetectionConfig]:
    '''Create sidebar widgets, apply pre-defined configs and return adjusted ones.'''
    st.sidebar.header('Select Options')

    available_modes = [f'{mode.name} ({mode.value})' for mode in RunningModes]
    mode = st.sidebar.radio('Running Mode', available_modes)
    mode = mode.split(' ')[0]

    st.sidebar.markdown("""---""")

    initial_config = default_config
    new_config = DetectionConfig()

    if mode == RunningModes.UPLOAD.name:
        uploaded_file = st.sidebar.file_uploader('Upload Image', ['png', 'jpg', 'jpeg'])

        img = format_uploaded_image(uploaded_file)

        # No image uploaded yet
        if img is None:
            return None, None

    elif mode == RunningModes.EXAMPLES.name:
        img_name = st.sidebar.selectbox('Input Image',
                                        list(choice.name for choice in IMAGES))
        if img_name == IMAGES.BIRDS.name:
            initial_config = birds_config
        elif img_name == IMAGES.DOTS.name:
            initial_config = dots_config
        elif img_name == IMAGES.BIRDS2.name:
            initial_config = birds2_config
        elif img_name == IMAGES.KEYBOARD.name:
            initial_config = keyboard_config
        img = load_image(getattr(IMAGES, img_name).value).copy()
        img = convert_color(img)

    new_config.thresh_val = st.sidebar.slider('Threshold Value',
                                              0,
                                              255,
                                              initial_config.thresh_val,
                                              help='Threshold Value, see [here](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57).')
    new_config.thresh_type = st.sidebar.slider('Threshold Type',
                                               0,
                                               5,
                                               initial_config.thresh_type,
                                               help='Threshold Operation Type, see [here](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576).')
    new_config.thresh_maxval = st.sidebar.slider('Thresh Max Val',
                                                 0,
                                                 255,
                                                 initial_config.thresh_maxval,
                                                 help='Maximum Value for Threshold Types 1 and 2, see [here](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57).')
    new_config.contour_mode = st.sidebar.slider('Contour Mode',
                                                0,
                                                3,
                                                initial_config.contour_mode,
                                                help='Contour Retrieval Mode, see [here](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71).')
    new_config.contour_method = st.sidebar.slider('Contour Method',
                                                  1,
                                                  4,
                                                  initial_config.contour_method,
                                                  help='Contour Approximation Method, see [here](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff).')

    new_config.min_contour_length = st.sidebar.slider(
        'Min Contour Length',
        min_value=0,
        max_value=100,
        value=initial_config.min_contour_length,
        help='Minimum Number of Vertices for Contour Graph.')
    new_config.max_contour_length = st.sidebar.slider(
        'Max Contour Length',
        min_value=0,
        max_value=100,
        value=initial_config.max_contour_length,
        help='Maximum Number of Vertices for Contour Graph.'
    )

    if st.sidebar.checkbox('Convert to grayscale?'):
        new_config.convert_grayscale = True
    return img, new_config


def main():
    st.set_page_config(page_title='Object Counter',
                       page_icon='🔢')
    img, config = create_sidebar()

    if img is None:
        return

    contours = find_contours(img,
                             thresh_val=config.thresh_val,
                             thresh_maxval=config.thresh_maxval,
                             thresh_type=config.thresh_type,
                             contour_mode=config.contour_mode,
                             contour_method=config.contour_method)
    n_objects, img = draw_contours(img,
                                   contours,
                                   min_contour_length=config.min_contour_length,
                                   max_contour_length=config.max_contour_length)

    st.subheader(f'Objects found: {n_objects}')

    if config.convert_grayscale:
        img = convert_gray(img)

    st.image(img,
             use_column_width=True)

if __name__=='__main__':
    main()
