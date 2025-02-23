#!/bin/bash

# Check if the file SciSciNet_Authors_Gender.tsv exists in the current directory
if [ ! -f "SciSciNet_Authors_Gender.tsv" ]; then
    echo "SciSciNet_Authors_Gender.tsv not found. Downloading..."
    wget -O SciSciNet_Authors_Gender.tsv https://springernature.figshare.com/ndownloader/files/38839116
else
    echo "SciSciNet_Authors_Gender.tsv already exists."
fi


# Check if the file SciSciNet_Papers.zip exists in the current directory
if [ ! -f "SciSciNet_Papers.zip" ]; then
    echo "SciSciNet_Papers.zip not found. Downloading..."
    wget -O SciSciNet_Papers.zip https://springernature.figshare.com/ndownloader/files/38839110
else
    echo "SciSciNet_Papers.zip already exists."
fi
echo "Unzipping SciSciNet_Papers.zip..."
unzip -n SciSciNet_Papers.zip


# Check if the file SciSciNet_Fields.tsv exists in the current directory
if [ ! -f "SciSciNet_Fields.tsv" ]; then
    echo "SciSciNet_Fields.tsv not found. Downloading..."
    wget -O SciSciNet_Fields.tsv https://springernature.figshare.com/ndownloader/files/36222114
else
    echo "SciSciNet_Fields.tsv already exists."
fi


# Check if the file SciSciNet_Authors.tsv exists in the current directory
if [ ! -f "SciSciNet_Authors.tsv" ]; then
    echo "SciSciNet_Authors.tsv not found. Downloading..."
    wget -O SciSciNet_Authors.tsv https://springernature.figshare.com/ndownloader/files/36139323
else
    echo "SciSciNet_Authors.tsv already exists."
fi

# Check if the file SciSciNet_PaperFields.tsv exists in the current directory
if [ ! -f "SciSciNet_PaperFields.tsv" ]; then
    echo "SciSciNet_PaperFields.tsv not found. Downloading..."
    wget -O SciSciNet_PaperFields.tsv https://springernature.figshare.com/ndownloader/files/36139311
else
    echo "SciSciNet_PaperFields.tsv already exists."
fi

# Check if the file SciSciNet_Link_Twitter.tsv exists in the current directory
if [ ! -f "SciSciNet_Link_Twitter.tsv" ]; then
    echo "SciSciNet_Link_Twitter.tsv not found. Downloading..."
    wget -O SciSciNet_Link_Twitter.tsv https://springernature.figshare.com/ndownloader/files/36139299
else
    echo "SciSciNet_Link_Twitter.tsv already exists."
fi


# Check if SciSciNet_Link_Patents.tsv exists in the current directory
if [ ! -f "SciSciNet_Link_Patents.tsv" ]; then
    echo "SciSciNet_Link_Patents.tsv not found. Downloading..."
    wget -O SciSciNet_Link_Patents.tsv https://springernature.figshare.com/ndownloader/files/36139293
else
    echo "SciSciNet_Link_Patents.tsv already exists."
fi