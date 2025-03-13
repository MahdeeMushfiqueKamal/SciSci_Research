## SciSciNet_Papers

134129189 lines

Description: 
```
PaperID (Integer) : Unique MAG Paper ID of the paper.
DOI (String) : Digital Object Identifier (DOI) of the paper.
DocType (String) : Book, BookChapter, Conference, Dataset, Journal, Repository, Thesis, or NULL (unknown).
Year (Integer) : Publication year of the paper.
Date (DateTime) : Publication date of the paper formatted as YYYY-MM-DD.
JournalID (Integer) : MAG Journal ID for published journal of the paper.
ConferenceSeriesID (Integer) : MAG ConferenceSeries ID for published conference series of the paper.
Reference_Count (Integer) : Total reference count of the paper.
Citation_Count (Integer) : Total citation count of the paper.
C5 (Integer) : The number of citations 5 years after publication.
C10 (Integer) : The number of citations 10 years after publication.
Disruption (Float) : Disruption score of the paper defined in Wu et al.
Atyp_Median_Z (Float) : Median Z-score of the paper defined in Uzzi et al.
Atyp_10pct_Z (Float) : 10th percentile Z-score of the paper defined in Uzzi et al.
Atyp_Pairs (Integer) : The number of journal pairs cite by the paper defined in Uzzi et al.
WSB_mu (Float) : Immediacy μ of the paper as introduced in WSB model.
WSB_sigma (Float) : Longevity σ of the paper as introduced in WSB model.
WSB_Cinf (Integer) : Ultimate impact of the paper predicted by WSB model.
SB_B (Float) : Beauty coefficient of the paper as introduced in Ke et al.
SB_T (Integer) : Awakening time of the paper as introduced in Ke et al.
Team_Size (Integer) : The number of researchers in the paper.
Institution_Count (Integer) : The number of institutions in the paper.
Patent_Count (Integer) : The number of citations by patents from USPTO and EPO.
Newsfeed_Count (Integer) : The number of mentions by news from Newsfeed.
Tweet_Count (Integer) : The number of mentions by tweets from Twitter.
NCT_Count (Integer) : The number of citations by clinical trials from ClinicalTrials.gov.
NIH_Count (Integer) : The number of supporting grants from NIH.
NSF_Count (Integer) : The number of supporting grants from NSF.
```