#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import required libraries
import xml.etree.cElementTree as ET
import os
import pprint

os.chdir("/Users/sarauenoyama/AnacondaProjects/Udacity_DAND/P4Project_Resources")

OSMFILE = "singapore.osm"
OSM_PATH = "singapore.osm"

# OSMFILE = "create_sample_singapore.osm"
# OSM_PATH = "create_sample_singapore.osm"

osm_file = open(OSMFILE, "r")


# find out what and how many tags there are
def count_tags(filename):
    # YOUR CODE HERE
    tags = {}
    for event, elem in ET.iterparse(filename):
        if elem.tag in tags.keys():
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
    return tags


tags = count_tags(OSMFILE)
pprint.pprint(tags)
