#!/bin/bash

curl -sS "https://files.de-1.osf.io/v1/resources/nuxr7/providers/osfstorage/?zip=" > dataset.zip
mkdir dataset
unzip dataset.zip -d dataset
rm dataset.zip

cd dataset
curl -sS https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5d4ec190265be2001830a77e/\?zip\= > things.zip
mkdir things
unzip things.zip -d things
rm things.zip

cd things/
mkdir object_images

DIRECTORY=.

for i in $DIRECTORY/object_images_*.zip; do
    unzip $i -d object_images
    rm -rf $i
done
