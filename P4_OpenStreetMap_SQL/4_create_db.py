#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import required libraries

import xml.etree.cElementTree as ET
import re
import csv
import codecs
import cerberus
import schema
import os
import sqlite3
import subprocess

## 1. Convert from XML to CSV
# I converted the XML file into CSV files with functions below, so that csv files can be easily imported to a SQL database.

os.chdir("/Users/sarauenoyama/AnacondaProjects/Udacity_DAND/P4Project_Resources")


OSMFILE = "singapore.osm"
OSM_PATH = "singapore.osm"
osm_file = open(OSMFILE, "r")
db_filename = 'singapore.db'

'''
OSMFILE = "create_sample_singapore.osm"
OSM_PATH = "create_sample_singapore.osm"
osm_file = open(OSMFILE, "r")
db_filename = 'sample_singapore.db'
'''

# the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

tree = ET.parse(osm_file)
root = tree.getroot()

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

cmd = "schema.py"
subprocess.Popen(cmd.split())

SCHEMA = schema

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = {}
    tags = {}  # Handle secondary tags the same way for both node and way elements

    # YOUR CODE HERE
    way_nodes = []
    tags = []
    if element.tag == 'node':
        
        for i in node_attr_fields:
            node_attribs[i] = element.attrib[i]
        
        for tag in element.iter('tag'):
            node_tags_attribs = {}
            temp = LOWER_COLON.search(tag.attrib['k'])
            is_p = PROBLEMCHARS.search(tag.attrib['k'])
            
            if is_p: # if the tag "k" value contains problematic characters, the tag should be ignored
                continue
            else:
                node_tags_attribs['id'] = element.attrib['id']
                node_tags_attribs['value'] = tag.attrib['v']
                
                if temp: # if the tag "k" value contains a ":" the characters before the ":" should be set as the tag type and characters after the ":" should be set as the tag key
                    split_char = temp.group(1)
                    split_index = tag.attrib['k'].index(split_char)
                    
                    node_tags_attribs['key'] = tag.attrib['k'][split_index+2:]
                    node_tags_attribs['type'] = tag.attrib['k'][:split_index+1]
                
                else: # if the tag "k" value does not contains a ":" "regular" should be set as the tag type
                    node_tags_attribs['key'] = tag.attrib['k']
                    node_tags_attribs['type'] = default_tag_type
            tags.append(node_tags_attribs)
            
       
        return {'node': node_attribs, 'node_tags': tags}
    
    elif element.tag == 'way':
        p = 0
        
        for i in way_attr_fields:
            way_attribs[i] = element.attrib[i]
            
        for tag2 in element.iter('tag'):
            way_tags_attribs = {}
            temp = LOWER_COLON.search(tag2.attrib['k'])
            is_p = PROBLEMCHARS.search(tag2.attrib['k'])
            
            if is_p: # if the tag "k" value contains problematic characters, the tag should be ignored
                continue
            else:
                way_tags_attribs['id'] = element.attrib['id']
                way_tags_attribs['value'] = tag2.attrib['v']
                
                if temp: # if the tag "k" value contains a ":" the characters before the ":" should be set as the tag type and characters after the ":" should be set as the tag key
                    split_char = temp.group(1)
                    split_index = tag2.attrib['k'].index(split_char)
                    
                    way_tags_attribs['key'] = tag2.attrib['k'][split_index+2:]
                    way_tags_attribs['type'] = tag2.attrib['k'][:split_index+1]
                
                else: # if the tag "k" value does not contains a ":" "regular" should be set as the tag type
                    way_tags_attribs['key'] = tag2.attrib['k']
                    way_tags_attribs['type'] = default_tag_type
            tags.append(way_tags_attribs)
        
        
        for j in element.iter('nd'):
            way_nodes_attribs = {}
            way_nodes_attribs['id'] = element.attrib['id']    
            way_nodes_attribs['node_id'] = j.attrib['ref']
            way_nodes_attribs['position'] = p
            p+=1
            
            way_nodes.append(way_nodes_attribs)
    
    
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}
    
# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()
            
def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))

class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
            
# ================================================== #
#               Main Function                                                        #
# ================================================== #

def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])

process_map(OSM_PATH, validate=True)


## 2. Create database from the supplied schema

schema_filename = 'data_wrangling_schema.sql'

delete_db = True
#FORCE DELETION of DB
if delete_db:
	if os.path.exists(db_filename):
		print ("FORCING DATABASE to be DELETED")
		os.remove(db_filename)

db_is_new = not os.path.exists(db_filename)
with sqlite3.connect(db_filename) as conn:
    if db_is_new:
        print 'Creating schema'
        with open(schema_filename, 'rt') as f:
            schema = f.read()
        conn.executescript(schema) # Create db from the supplied schema
    else:
        print 'Database exists, assume schema does, too.'
conn.close()

## 3. Import CSV into SQL database

data_filename = ['nodes.csv', 'nodes_tags.csv', 'ways.csv', 'ways_tags.csv', 'ways_nodes.csv']

# connect to singapore.db
conn = sqlite3.connect("singapore.db")
cursor = conn.cursor()

# create tables on singapore.db
# Dictionary of instructions per file
# "OR IGNORE" : to avoid: sqlite3.IntegrityError: UNIQUE constraint failed: 

SQL = {}
SQL["nodes.csv"] = """INSERT OR IGNORE INTO nodes (id, lat, lon, user, uid, version, changeset, timestamp) values (:id, :lat, :lon, :user, :uid, :version, :changeset, :timestamp)"""

#nodes_tags
SQL["nodes_tags.csv"] = """INSERT OR IGNORE INTO nodes_tags (id, key, value, type) values (:id, :key, :value, :type) """

#ways
SQL["ways.csv"] = """INSERT OR IGNORE INTO ways (id, user, uid, version, changeset, timestamp) values (:id, :user, :uid, :version, :changeset, :timestamp)  """

#ways_tags
SQL["ways_tags.csv"] = """INSERT OR IGNORE INTO ways_tags (id, key, value, type) values (:id, :key, :value, :type) """

#ways_nodes
SQL["ways_nodes.csv"] = """INSERT OR IGNORE INTO ways_nodes (id, node_id, position) values (:id, :node_id, :position)  """


for x in data_filename:
	print x
	with open(x, 'rt') as csv_file:
	    csv_reader = csv.DictReader(csv_file)
	    print SQL[x]
	    
	    with sqlite3.connect(db_filename) as conn:
	    	conn.text_factory = str
	        cursor = conn.cursor()
	        cursor.executemany(SQL[x], csv_reader)
	        conn.commit()
	    conn.close()