#!/bin/bash

# Check if the file SciSciNet_Authors_Gender.tsv exists in the current directory
if [ ! -f "SciSciNet_Authors_Gender.tsv" ]; then
    echo "SciSciNet_Authors_Gender.tsv not found. Downloading..."
    wget -O SciSciNet_Authors_Gender.tsv https://springernature.figshare.com/ndownloader/files/38839116
else
    echo "SciSciNet_Authors_Gender.tsv already exists."
fi


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


# Check if the file SciSciNet_NSF_Metadata.tsv exists in the current directory
if [ ! -f "SciSciNet_NSF_Metadata.tsv" ]; then
    echo "SciSciNet_NSF_Metadata.tsv not found. Downloading..."
    wget -O SciSciNet_NSF_Metadata.tsv https://springernature.figshare.com/ndownloader/files/36139296
else
    echo "SciSciNet_NSF_Metadata.tsv already exists."
fi


# Check if the file SciSciNet_Newsfeed_Metadata.tsv exists in the current directory
if [ ! -f "SciSciNet_Newsfeed_Metadata.tsv" ]; then
    echo "SciSciNet_Newsfeed_Metadata.tsv not found. Downloading..."
    wget -O SciSciNet_Newsfeed_Metadata.tsv https://springernature.figshare.com/ndownloader/files/36139290
else
    echo "SciSciNet_Newsfeed_Metadata.tsv already exists."
fi


# Check if the file SciSciNet_Papers.tsv exists in the current directory
if [ ! -f "SciSciNet_Papers.tsv" ]; then
    echo "SciSciNet_Papers.tsv not found. Downloading..."
    wget -O SciSciNet_Papers.tsv https://springernature.figshare.com/ndownloader/files/36139287
else
    echo "SciSciNet_Papers.tsv already exists."
fi


# Check if the file SciSciNet_PaperAuthorAffiliations.tsv exists in the current directory
if [ ! -f "SciSciNet_PaperAuthorAffiliations.tsv" ]; then
    echo "SciSciNet_PaperAuthorAffiliations.tsv not found. Downloading..."
    wget -O SciSciNet_PaperAuthorAffiliations.tsv https://springernature.figshare.com/ndownloader/files/36139278
else
    echo "SciSciNet_PaperAuthorAffiliations.tsv already exists."
fi


# Check if the file SciSciNet_Twitter_Metadata.tsv exists in the current directory
if [ ! -f "SciSciNet_Twitter_Metadata.tsv" ]; then
    echo "SciSciNet_Twitter_Metadata.tsv not found. Downloading..."
    wget -O SciSciNet_Twitter_Metadata.tsv https://springernature.figshare.com/ndownloader/files/36139260
else
    echo "SciSciNet_Twitter_Metadata.tsv already exists."
fi


# Check if the file SciSciNet_Link_NIH.tsv exists in the current directory
if [ ! -f "SciSciNet_Link_NIH.tsv" ]; then
    echo "SciSciNet_Link_NIH.tsv not found. Downloading..."
    wget -O SciSciNet_Link_NIH.tsv https://springernature.figshare.com/ndownloader/files/36139254
else
    echo "SciSciNet_Link_NIH.tsv already exists."
fi


# Check if the file SciSciNet_Link_Newsfeed.tsv exists in the current directory
if [ ! -f "SciSciNet_Link_Newsfeed.tsv" ]; then
    echo "SciSciNet_Link_Newsfeed.tsv not found. Downloading..."
    wget -O SciSciNet_Link_Newsfeed.tsv https://springernature.figshare.com/ndownloader/files/36139251
else
    echo "SciSciNet_Link_Newsfeed.tsv already exists."
fi


# Check if the file SciSciNet_Affiliations.tsv exists in the current directory
if [ ! -f "SciSciNet_Affiliations.tsv" ]; then
    echo "SciSciNet_Affiliations.tsv not found. Downloading..."
    wget -O SciSciNet_Affiliations.tsv https://springernature.figshare.com/ndownloader/files/36139245
else
    echo "SciSciNet_Affiliations.tsv already exists."
fi


# Check if the file SciSciNet_Journals.tsv exists in the current directory
if [ ! -f "SciSciNet_Journals.tsv" ]; then
    echo "SciSciNet_Journals.tsv not found. Downloading..."
    wget -O SciSciNet_Journals.tsv https://springernature.figshare.com/ndownloader/files/36139248
else
    echo "SciSciNet_Journals.tsv already exists."
fi


# Check if the file SciSciNet_Link_ClinicalTrials.tsv exists in the current directory
if [ ! -f "SciSciNet_Link_ClinicalTrials.tsv" ]; then
    echo "SciSciNet_Link_ClinicalTrials.tsv not found. Downloading..."
    wget -O SciSciNet_Link_ClinicalTrials.tsv https://springernature.figshare.com/ndownloader/files/36139236
else
    echo "SciSciNet_Link_ClinicalTrials.tsv already exists."
fi


# Check if the file SciSciNet_Link_NSF.tsv exists in the current directory
if [ ! -f "SciSciNet_Link_NSF.tsv" ]; then
    echo "SciSciNet_Link_NSF.tsv not found. Downloading..."
    wget -O SciSciNet_Link_NSF.tsv https://springernature.figshare.com/ndownloader/files/36139242
else
    echo "SciSciNet_Link_NSF.tsv already exists."
fi


# Check if the file SciSciNet_ConferenceSeries.tsv exists in the current directory
if [ ! -f "SciSciNet_ConferenceSeries.tsv" ]; then
    echo "SciSciNet_ConferenceSeries.tsv not found. Downloading..."
    wget -O SciSciNet_ConferenceSeries.tsv https://springernature.figshare.com/ndownloader/files/36139239
else
    echo "SciSciNet_ConferenceSeries.tsv already exists."
fi


# Check if the file SciSciNet_Link_NobelLaureates.tsv exists in the current directory
if [ ! -f "SciSciNet_Link_NobelLaureates.tsv" ]; then
    echo "SciSciNet_Link_NobelLaureates.tsv not found. Downloading..."
    wget -O SciSciNet_Link_NobelLaureates.tsv https://springernature.figshare.com/ndownloader/files/36139230
else
    echo "SciSciNet_Link_NobelLaureates.tsv already exists."
fi

# ####################### zip files ############################

# Check if the file SciSciNet_Papers.zip exists in the current directory
if [ ! -f "SciSciNet_Papers.zip" ]; then
    echo "SciSciNet_Papers.zip not found. Downloading..."
    wget -O SciSciNet_Papers.zip https://springernature.figshare.com/ndownloader/files/38839110
else
    echo "SciSciNet_Papers.zip already exists."
fi
echo "Unzipping SciSciNet_Papers.zip..."
unzip -n SciSciNet_Papers.zip


# Check if the file SciSciNet_PaperDetails.tsv.zip exists in the current directory
if [ ! -f "SciSciNet_PaperDetails.tsv.zip" ]; then
    echo "SciSciNet_PaperDetails.tsv.zip not found. Downloading..."
    wget -O SciSciNet_PaperDetails.tsv.zip https://springernature.figshare.com/ndownloader/files/36139233
else
    echo "SciSciNet_PaperDetails.tsv.zip already exists."
fi
echo "Unzipping SciSciNet_PaperDetails.tsv.zip..."
unzip -n SciSciNet_PaperDetails.tsv.zip


# Check if the file SciSciNet_PaperReferences.tsv.zip exists in the current directory
if [ ! -f "SciSciNet_PaperReferences.tsv.zip" ]; then
    echo "SciSciNet_PaperReferences.tsv.zip not found. Downloading..."
    wget -O SciSciNet_PaperReferences.tsv.zip https://springernature.figshare.com/ndownloader/files/36139221
else
    echo "SciSciNet_PaperReferences.tsv.zip already exists."
fi
echo "Unzipping SciSciNet_PaperReferences.tsv.zip..."
unzip -n SciSciNet_PaperReferences.tsv.zip