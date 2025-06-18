# This script handles the download of the geospatial data, hosted in the JRC Data Catalogue.

baseurl="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/PVGIS"
subdir="climatic-zones-pv"

echo "Downloading copyright notice"
wget "$baseurl/copyright.txt" -O "copyright.txt"

mkdir -p "$subdir"

for file in  "dataset_description.md" "means-global.csv" "means-europe.csv" "icdf-daily-G-europe.pickle" "icdf-daily-G-global.pickle" "icdf-daily-suff-europe.pickle" "icdf-daily-suff-global.pickle"; 
do
    echo "Downloading $file"
    wget "$baseurl/$subdir/$file" -O "$subdir/$file"
done
