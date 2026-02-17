from pydeck import View
import pandas as pd
import json
import src.io_utils as iou


file_path = "d:/GS_LCR_DELIVERABLE/test/test.json"
out_file = "d:/GS_LCR_DELIVERABLE/test/test.json"
trees = iou.load_project_json(file_path)
