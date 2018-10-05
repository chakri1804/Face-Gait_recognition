import glob, os
import pandas as pd
import numpy as np
import cv2
from lxml import etree
import xml.etree.cElementTree as ET

path = 'test/'
erects = pd.read_csv('bb_landmark/loose_bb_train.csv')
blah = np.asarray(erects)

def write_xml(folder, img, objects,tl, br,savedir):
	if not os.path.isdir(savedir):
		os.mkdir(savedir)

	annotation = ET.Element('annotation')
	ET.SubElement(annotation, 'folder').text = folder
	ET.SubElement(annotation, 'filename').text = img+'.jpg'
	ET.SubElement(annotation, 'segmented').text = '0'
	size = ET.SubElement(annotation, 'size')
	ET.SubElement(size, 'width').text = '80'
	ET.SubElement(size, 'height').text = '100'
	ET.SubElement(size, 'depth').text = '3'
	for obj, topl, botr in zip(objects, tl, br):
		ob = ET.SubElement(annotation, 'object')
		ET.SubElement(ob, 'name').text = obj
		ET.SubElement(ob, 'pose').text = 'Unspecified'
		ET.SubElement(ob, 'truncated').text = '0'
		ET.SubElement(ob, 'difficult').text = '0'
		bbox = ET.SubElement(ob, 'bndbox')
		ET.SubElement(bbox, 'xmin').text = str(topl[0])
		ET.SubElement(bbox, 'ymin').text = str(botr[1])
		ET.SubElement(bbox, 'xmax').text = str(botr[0])
		ET.SubElement(bbox, 'ymax').text = str(topl[1])

	xml_str = ET.tostring(annotation)
	root = etree.fromstring(xml_str)
	xml_str = etree.tostring(root, pretty_print=True)
	save_path = os.path.join(savedir,folder+'_'+img+'.xml')
	with open(save_path, 'wb') as temp_xml:
		temp_xml.write(xml_str)

for pic, x, y, w, h in blah:
	if(x>0 and y>0):
		print(pic)
		folder = path+pic[0:7]
		img = pic[8:]
		objects = pic[0:7]
		tl = [(0,100)]
		br = [(80,0)]
		savedir = 'annotation'
		write_xml(folder, img, objects, tl, br, savedir)
		# os.system('mkdir '+bad+pic[0:8])
		# os.system('mv '+path+pic+'.jpg '+bad+pic+'.jpg')
