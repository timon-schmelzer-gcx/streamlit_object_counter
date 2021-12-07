import typing

import cv2
import numpy as np
from numpy.lib.polynomial import poly
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

def filter_contours(contours: typing.Tuple,
                    min_contour_length: int=-9999,
                    max_contour_length: int=9999) -> np.array:
    '''Filter contours by given criteria.'''
    return np.array([
        contour for contour in contours
        if len(contour) >= min_contour_length and len(contour) <= max_contour_length
    ], dtype=object)


def draw_contours(img: np.array,
                  contours: np.array,
                  contour_index: int=-1,
                  contour_color: typing.Tuple=(255, 0, 0, 1),
                  contour_thickness: int=0) -> typing.Tuple[int, typing.Tuple]:
    '''Add contours if they fullfill the given criteria.'''
    n_contours = 0
    for contour in contours:
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

def polygon_sizes(contours: list) -> list[float]:
    sizes = []
    for contour in contours:
        xs = np.array(contour, dtype=object)[:,0,0]
        ys = np.array(contour, dtype=object)[:,0,1]
        sizes.append(calculate_ploygon_size(xs, ys))
    return np.array(sizes)

def calculate_ploygon_size(xs: np.array, ys: np.array):
    return 0.5*np.abs(np.dot(xs,np.roll(ys,1))-np.dot(ys,np.roll(xs,1)))

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

    min_contour_length, max_contour_length = st.sidebar.slider(
        'Min/Max Contour Length',
        min_value=0,
        max_value=100,
        value=(initial_config.min_contour_length, initial_config.max_contour_length),
        help='Number of Vertices for Contour Graph.')

    new_config.min_contour_length = min_contour_length
    new_config.max_contour_length = max_contour_length

    return img, new_config

def plot_histogram(data: np.array) -> go.Figure:
    '''Plot histogram of given data.'''
    fig = px.histogram(x=data)
    fig.update_layout(
        xaxis_title='Polygon Areas',
        yaxis_title='Polygon Sizes'
    )
    return fig

def main():
    st.set_page_config(page_title='Object Counter',
                       layout='wide',
                       page_icon='🔢')
    st.markdown('''
       # Object Counter App
       This app helps you to count similar shaped objects on a given image. To improve
       the object detection performance, you can adjust the parameters on the left hand side.
       Look at the example to get a feeling for that. Afterwards, upload an image and try it
       yourself. Good luck 🍀!
    ''')
    img, config = create_sidebar()

    if img is None:
        return

    contours = find_contours(img,
                             thresh_val=config.thresh_val,
                             thresh_maxval=config.thresh_maxval,
                             thresh_type=config.thresh_type,
                             contour_mode=config.contour_mode,
                             contour_method=config.contour_method)

    contours_filtered = filter_contours(contours,
                                        min_contour_length=config.min_contour_length,
                                        max_contour_length=config.max_contour_length)

    # refilter contours by polygon size
    polygons = polygon_sizes(contours_filtered)
    min_area, max_area = st.sidebar.slider('Polygon Size',
                                           value=(int(min(polygons)), int(max(polygons))+1),
                                           min_value=int(min(polygons)),
                                           max_value=int(max(polygons))+1)
    contour_indices = np.argwhere((polygons >= min_area) & (polygons <= max_area))
    contours_refiltered = []
    polygons_filtered = []
    for i, (polygon, contour) in enumerate(zip(polygons, contours_filtered)):
        if i in contour_indices:
            contours_refiltered.append(contour)
            polygons_filtered.append(polygon)

    n_objects, img = draw_contours(img,
                                   contours_refiltered)

    n_objects, img_contours = draw_contours(np.zeros(shape=img.shape) + 255,
                                            contours_refiltered)

    st.subheader(f'Objects found: {n_objects}')
    fig = plot_histogram(polygons_filtered)

    col_one, col_two, col_three = st.columns((1, 1, 1))
    col_one.image(img,
                caption='Original image with contours',
                use_column_width=True)
    col_two.image(img_contours,
                caption='Contours',
                clamp=True,
                use_column_width=True)
    col_three.markdown('If the objects are equal in size, the polygon areas '
                       'should follow a normal distribution. Modify the "Polygon Size" '
                       'attribute to filter out outliers.')
    col_three.plotly_chart(fig)

if __name__=='__main__':
    main()
