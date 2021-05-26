from pathlib import Path
import os
# pylint: disable=line-too-long

################################################################################
################################################################################
##                      ####       #      #####    #   #                      ##
##                      #   #     # #       #      #   #                      ##
##                      ####     #####      #      #####                      ##
##                      #        #   #      #      #   #                      ##
##                      #        #   #      #      #   #                      ##
################################################################################
################################################################################


MODULE_PATH           = (Path(__file__) / '../').resolve()

# dev
CROP__RESULT__DIR            = f"{MODULE_PATH}/results/crop/result"
CROP__VERSIONS__DIR          = f"{MODULE_PATH}/results/crop/versions"
CROP__BNA__DIR               = f"{MODULE_PATH}/results/crop/bna"
################################################################################
SEGMENT_COLOR__RESULT__DIR   = f"{MODULE_PATH}/results/color_segment/result"
SEGMENT_COLOR__VERSIONS__DIR = f"{MODULE_PATH}/results/color_segment/versions"
SEGMENT_COLOR__BNA__DIR      = f"{MODULE_PATH}/results/color_segment/bna"
################################################################################
SEGMENT_WALL__MASK__DIR    = f"{MODULE_PATH}/results/wall_segment/mask"
# SEGMENT_WALL__RESULT__DIR    = f"{MODULE_PATH}/results/wall_segment/result"
# SEGMENT_WALL__VERSIONS__DIR  = f"{MODULE_PATH}/results/wall_segment/versions"
# SEGMENT_WALL__BNA__DIR       = f"{MODULE_PATH}/results/wall_segment/bna"
################################################################################
OCR__BOXES__DIR              = f"{MODULE_PATH}/results/ocr/boxes"
OCR__PICKLE__DIR             = f"{MODULE_PATH}/results/ocr/pickle"
################################################################################
REMOVE_TEXT__RESULT__DIR     = f"{MODULE_PATH}/results/remove_text/result"
REMOVE_TEXT__VERSIONS__DIR   = f"{MODULE_PATH}/results/remove_text/versions"
REMOVE_TEXT__BNA__DIR        = f"{MODULE_PATH}/results/remove_text/bna"
################################################################################
REMOVE_CIRCLE__RESULT__DIR   = f"{MODULE_PATH}/results/remove_circle/result"
REMOVE_CIRCLE__VERSIONS__DIR = f"{MODULE_PATH}/results/remove_circle/versions"
REMOVE_CIRCLE__BNA__DIR      = f"{MODULE_PATH}/results/remove_circle/bna"
################################################################################
REMOVE_ICON__RESULT__DIR     = f"{MODULE_PATH}/results/remove_icon/result"
REMOVE_ICON__VERSIONS__DIR   = f"{MODULE_PATH}/results/remove_icon/versions"
REMOVE_ICON__BNA__DIR        = f"{MODULE_PATH}/results/remove_icon/bna"
DETECT_ICON__WEIGHT__PATH    = f"{MODULE_PATH}/detect_icon/weights/best.pt"
################################################################################
WALL_SEGMENT__WEIGHT__PATH   = f"{MODULE_PATH}/wall_segmentation/model/frozen_inference_graph.pb"
################################################################################
ADD_TEXT__RESULT__DIR        = f"{MODULE_PATH}/results/add_text/result"
ADD_TEXT__VERSIONS__DIR      = f"{MODULE_PATH}/results/add_text/versions"
ADD_TEXT__BNA__DIR           = f"{MODULE_PATH}/results/add_text/bna"
################################################################################
CLEAN_FINAL__RESULT__DIR     = f"{MODULE_PATH}/results/clean_final/result"
CLEAN_FINAL__VERSIONS__DIR   = f"{MODULE_PATH}/results/clean_final/versions"
CLEAN_FINAL__BNA__DIR        = f"{MODULE_PATH}/results/clean_final/bna"
################################################################################
FINAL_RESULT__RESULT__DIR    = f"{MODULE_PATH}/results/final_result/result"
FINAL_RESULT__BNA__DIR       = f"{MODULE_PATH}/results/final_result/bna"
FINAL_RESULT__STEPS__DIR     = f"{MODULE_PATH}/results/final_result/steps"

# webservice
UPLOAD__DIR                  = f"{MODULE_PATH}/webservice/static/uploads/"
OUTPUT__DIR                  = f"{MODULE_PATH}/webservice/static/output/"

# log
BUG__LOG__PATH               = f"{MODULE_PATH}/logs/bugs_ignore.log"

# Log time
LOG_TIME                     = True


# Auto makedirs
paths = [
    CROP__RESULT__DIR, CROP__VERSIONS__DIR, CROP__BNA__DIR,
    SEGMENT_COLOR__RESULT__DIR, SEGMENT_COLOR__VERSIONS__DIR, SEGMENT_COLOR__BNA__DIR,
    # TODO: add segment_wall
    OCR__BOXES__DIR, OCR__PICKLE__DIR,
    REMOVE_TEXT__RESULT__DIR, REMOVE_TEXT__VERSIONS__DIR, REMOVE_TEXT__BNA__DIR,
    REMOVE_CIRCLE__RESULT__DIR, REMOVE_CIRCLE__VERSIONS__DIR, REMOVE_CIRCLE__BNA__DIR,
    REMOVE_ICON__RESULT__DIR, REMOVE_ICON__VERSIONS__DIR, REMOVE_ICON__BNA__DIR,
    ADD_TEXT__RESULT__DIR, ADD_TEXT__VERSIONS__DIR, ADD_TEXT__BNA__DIR,
    CLEAN_FINAL__RESULT__DIR, CLEAN_FINAL__VERSIONS__DIR, CLEAN_FINAL__BNA__DIR,
    FINAL_RESULT__BNA__DIR, FINAL_RESULT__RESULT__DIR, FINAL_RESULT__STEPS__DIR,
    UPLOAD__DIR, OUTPUT__DIR,
    ]

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
        # create .gitkeep
        # with open(os.path.join(path,'.gitkeep'), 'w'):
        #     pass


################################################################################
################################################################################
##                       ###     #####    #####    ####                       ##
##                      #          #      #        #   #                      ##
##                       ###       #      #####    ####                       ##
##                          #      #      #        #                          ##
##                       ###       #      #####    #                          ##
################################################################################
################################################################################


steps_paths = { # unordered
    'crop'          : CROP__RESULT__DIR,
    'segment_color' : SEGMENT_COLOR__RESULT__DIR,
    'ocr'           : OCR__BOXES__DIR,
    'remove_text'   : REMOVE_TEXT__RESULT__DIR,
    'remove_circle' : REMOVE_CIRCLE__RESULT__DIR,
    'remove_icon'   : REMOVE_ICON__RESULT__DIR,
    'clean_final'   : CLEAN_FINAL__RESULT__DIR,
    'add_text'      : ADD_TEXT__RESULT__DIR,
    'final_result'  : FINAL_RESULT__RESULT__DIR,
}

################################################################################
################################################################################
##      ###  ####  #####       ####  #####  #### ##### #####  #### #####      ##
##     #   # #   #   #         #   # #     #       #   #     #       #        ##
##     #   # ####    #         #   # ##### #       #   ##### #       #        ##
##     #   # #   #   #         #   # #     #       #   #     #       #        ##
##      ###  ####  ##          ####  #####  ####   #   #####  ####   #        ##
################################################################################
################################################################################

CLASSES = ['cir_tri', 'circle_1', 'circle_2', 'circle_3', 'circle_4',
           'circle_5', 'circle_alpha', 'door', 'rect_1', 'rect_2', 'rect_3',
           'rect_4', 'triangle']

INPUT_SIZE = (640, 640)