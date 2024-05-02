# !/bin/sh

for f in $(ls ~/Desktop/AI\ in\ Medical\ Imaging/Project/annotations/ULS23/processed_data/fully_annotated); do
    cd ~/Desktop/AI\ in\ Medical\ Imaging/Project/annotations/ULS23/processed_data/fully_annotated
    echo "Processing $f"
    if test -d "$f/labelsTr"; then
        echo "Labels already processed"
        continue
    fi
    if ! test -d "$f/labels"; then
        echo "Labels not found"
        continue
    fi
    cd ~/Desktop/AI\ in\ Medical\ Imaging/Project/annotations/ULS23/processed_data/fully_annotated/$f/labels
    mkdir -p ../labelsTr
    for i in *.zip; do 
        unzip $i -d ../labelsTr
    done
done