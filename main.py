import os
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import widgets as wid
import resources as res
import samdetector as sam
import numpy as np

class IamSamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("I Am Sam")

        basepath = os.path.dirname(__file__)
        basename = 'main'
        uifile = os.path.join(basepath, '%s.ui' % basename)
        wid.loadUi(uifile, self)

        # add actions to action group
        ag = QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.actionPlusPoint)
        ag.addAction(self.actionHand_selector)
        ag.addAction(self.actionNegPoint)

        self.viewer = wid.PhotoViewer(self)
        self.verticalLayout.addWidget(self.viewer)

        # initial parameters
        self.list_models = ['FASTSAM', 'SAM']
        self.comboBox.addItems(self.list_models)
        self.active_model = self.list_models[0]

        self.actionLoadImage.triggered.connect(self.get_image)
        self.actionNegPoint.triggered.connect(self.add_neg_point)
        self.actionPlusPoint.triggered.connect(self.add_plus_point)
        self.actionSegment.triggered.connect(self.go_segment)
        self.actionRemovePoints.triggered.connect(self.reset_points)

        self.viewer.end_pluspoint_selection.connect(self.plus_point_added)
        self.viewer.end_minpoint_selection.connect(self.min_point_added)
        self.comboBox.currentIndexChanged.connect(self.update_model)

        self.add_icon(res.find('img/photo.png'), self.actionLoadImage)
        self.add_icon(res.find('img/plus.png'), self.actionPlusPoint)
        self.add_icon(res.find('img/min.png'), self.actionNegPoint)
        self.add_icon(res.find('img/magic.png'), self.actionSegment)
        self.add_icon(res.find('img/hand.png'), self.actionHand_selector)
        self.add_icon(res.find('img/reset.png'), self.actionRemovePoints)

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QIcon(img_source))
    def go_segment(self):
        # get type of model to use
        plus_pts = self.viewer.list_point_plus
        min_pts = self.viewer.list_point_min

        glob_list = plus_pts + min_pts
        n_first_elements = [1] * len(plus_pts)
        m_next_elements = [0] * len(min_pts)
        labels = n_first_elements + m_next_elements
        print(labels)

        if self.active_model == 'FASTSAM':

            # launch FastSAM
            p, ann = sam.do_fast_sam(self.image_path, glob_list, labels)
            output_path = 'results_fastsam.jpg'
            sam.fastsam_create_mask_image(ann, output_path)

        else:
            masks, scores = sam.do_sam(self.image_path, np.array(glob_list), np.array(labels))
            output_path = 'results_sam.jpg'
            sam.sam_create_mask_image(masks[0], output_path)

        self.viewer.compose_mask_image(output_path)

    def update_model(self):
        i = self.comboBox.currentIndex()
        self.active_model = self.list_models[i]

    def reset_points(self):
        self.viewer.clean_scene()
        self.viewer.list_point_plus = []
        self.viewer.list_point_min = []
        self.viewer.pluspoint_count = 0
        self.viewer.minpoint_count = 0

    def add_neg_point(self):
        if self.actionNegPoint.isChecked():
            self.viewer.select_point_min = True
            self.viewer.toggleDragMode()

    def add_plus_point(self):
        if self.actionPlusPoint.isChecked():
            self.viewer.select_point_plus = True
            self.viewer.toggleDragMode()

    def plus_point_added(self):
        interest_point = self.viewer._current_point
        x = interest_point.x()
        y = interest_point.y()
        print(f'Added Pos point: {x},{y}')
        self.hand_pan()

        if self.viewer.pluspoint_count > 0:
            self.actionSegment.setEnabled(True)

    def min_point_added(self):
        interest_point = self.viewer._current_point
        x = interest_point.x()
        y = interest_point.y()
        print(f'Added Pos point: {x},{y}')
        self.hand_pan()

    def reset_parameters(self):
        pass
    def get_image(self):
        """
        Get the image path from the user
        :return:
        """
        try:
            img = QFileDialog.getOpenFileName(self, u"Ouverture de fichiers", "",
                                                        "Image Files (*.png *.jpg *.bmp *.tif)")
            print(f'the following image will be loaded {img[0]}')
        except:
            pass
        if img[0] != '':
            # load and show new image
            self.load_image(img[0])

    def load_image(self, path):
        """
        Load the new image and reset the model
        :param path:
        :return:
        """
        self.reset_points()

        self.image_path = path
        self.viewer.setPhoto(QPixmap(path))
        self.viewer.set_base_image(path)
        self.image_loaded = True

        # enable action
        self.actionPlusPoint.setEnabled(True)
        self.actionNegPoint.setEnabled(True)
        self.actionHand_selector.setChecked(True)

    def hand_pan(self):
        # switch back to hand tool
        self.actionHand_selector.setChecked(True)



# Press the green button in the gutter to run the script.
def main(argv=None):
    """
    Creates the main window for the application and begins the \
    QApplication if necessary.

    :param      argv | [, ..] || None

    :return      error code
    """

    # Define installation path
    install_folder = os.path.dirname(__file__)

    app = None

    # create the application if necessary
    if (not QApplication.instance()):
        app = QApplication(argv)
        app.setStyle('Fusion')

    # create the main window

    window = IamSamApp()
    window.showMaximized()

    # run the application if necessary
    if (app):
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))