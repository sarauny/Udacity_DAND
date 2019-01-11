#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import required libraries

import os
import xml.etree.cElementTree as ET
import re
import pprint

os.chdir("/Users/sarauenoyama/AnacondaProjects/Udacity_DAND/P4Project_Resources")

OSMFILE = "singapore.osm"
OSM_PATH = "singapore.osm"

# OSMFILE = "create_sample_singapore.osm"
# OSM_PATH = "create_sample_singapore.osm"

osm_file = open(OSMFILE, "r")

# 3 regular expressions to check patterns in the tags
lower = re.compile(r'^([a-z]|_)*$')  # those contain only lowercase letters
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')  # tags with a colon in their names
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')  # tags with problematic characters


# function devides tags into three categories
def key_type(element, keys):
    if element.tag == "tag":
        # YOUR CODE HERE
        if element.tag == "tag":
            if lower.search(element.attrib['k']):
                keys['lower'] += 1
            elif lower_colon.search(element.attrib['k']):
                keys['lower_colon'] += 1
            elif problemchars.search(element.attrib['k']):
                keys['problemchars'] += 1
            else:
                keys['other'] += 1

    return keys


def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys


keys = process_map(OSMFILE)
pprint.pprint(keys)
